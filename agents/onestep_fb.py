import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue


class OneStepFBAgent(flax.struct.PyTreeNode):
    """One-step forward-backward representation learning (one-step FB) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def fb_repr_loss(self, batch, grad_params):
        """Compute the forward backward representation loss."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        # Compute target successor measures.
        target_next_forward_reprs = self.network.select('target_forward_repr')(
            next_observations, actions=next_actions)
        target_backward_reprs = self.network.select('target_backward_repr')(
            observations, actions=actions)
        target_succ_measures = jnp.einsum(
            'eij,kj->eik',
            target_next_forward_reprs,
            target_backward_reprs,
        )
        if self.config['repr_agg'] == 'mean':
            target_succ_measures = jnp.mean(target_succ_measures, axis=0)
        else:
            target_succ_measures = jnp.min(target_succ_measures, axis=0)

        # Compute successor measures.
        forward_reprs = self.network.select('forward_repr')(
            observations, actions=actions, params=grad_params)
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions, params=grad_params)
        succ_measures = jnp.einsum('eij,kj->eik', forward_reprs, backward_reprs)

        # Compute the TD LSIF loss.
        I = jnp.eye(batch_size)
        repr_off_diag_loss = jax.vmap(
            lambda x: (x * (1 - I)) ** 2,
            0, 0
        )(succ_measures - self.config['discount'] * target_succ_measures[None])
        repr_off_diag_loss = 0.5 * jnp.sum(repr_off_diag_loss, axis=-1) / (batch_size - 1)
        repr_off_diag_loss = jnp.mean(repr_off_diag_loss)

        repr_diag_loss = -(1 - self.config['discount']) * jax.vmap(jnp.diag, 0, 0)(succ_measures)
        repr_diag_loss = jnp.mean(repr_diag_loss)

        repr_loss = repr_diag_loss + repr_off_diag_loss

        # Compute orthonormalization regularization.
        covariance = jnp.matmul(backward_reprs, backward_reprs.T)
        ortho_diag_loss = -jnp.diag(covariance).mean()
        ortho_off_diag_loss = 0.5 * jnp.sum((covariance * (1 - I)) ** 2, axis=-1) / (batch_size - 1)
        ortho_off_diag_loss = jnp.mean(ortho_off_diag_loss)
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss

        fb_loss = repr_loss + self.config['orthonorm_coeff'] * ortho_loss

        return fb_loss, {
            'fb_loss': fb_loss,
            'repr_loss': repr_loss,
            'repr_diag_loss': repr_diag_loss,
            'repr_off_diag_loss': repr_off_diag_loss,
            'ortho_loss': ortho_loss,
            'ortho_diag_loss': ortho_diag_loss,
            'ortho_off_diag_loss': ortho_off_diag_loss,
            'succ_measure_mean': succ_measures.mean(),
            'succ_measure_max': succ_measures.max(),
            'succ_measure_min': succ_measures.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the RPG+BC actor loss."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']

        # Sample actions.
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        forward_reprs = self.network.select('forward_repr')(
            observations, actions=q_actions)
        qs = jnp.einsum('eik,ik->ei', forward_reprs, latents)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        # Compute BC loss.
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss = q_loss + self.config['alpha'] * bc_loss
        
        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - actions) ** 2),
            'std': action_std,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, latent_rng, actor_rng = jax.random.split(rng, 3)

        # Sample latents.
        batch['latents'] = self.sample_latents(batch, latent_rng)

        # Train the one-step FB representations.
        fb_repr_loss, fb_repr_info = self.fb_repr_loss(batch, grad_params)
        for k, v in fb_repr_info.items():
            info[f'fb_repr/{k}'] = v

        # Train the actor to maximize the inner products.
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = fb_repr_loss + actor_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'forward_repr')
        self.target_update(new_network, 'backward_repr')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        """Infer the latent variable using rewards on downstream tasks."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        weights = jax.nn.softmax(self.config['reward_temperature'] * rewards, axis=0)
        
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions)

        # reward-weighted average
        latent = jnp.mean((weights * rewards)[..., None] * backward_reprs, axis=0)
        if self.config['normalize_latent']:
            latent = latent / jnp.linalg.norm(
                latent, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Sample latent variables and intrinsic rewards."""
        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']

        rng, latent_rng, perm_rng, mix_rng = jax.random.split(rng, 4)
        
        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        if self.config['normalize_latent']:
            latents = latents / jnp.linalg.norm(
                latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        
        perm = jax.random.permutation(perm_rng, jnp.arange(batch_size))
        backward_reprs = self.network.select('backward_repr')(
            observations, actions=actions)
        latent_backward_reprs = backward_reprs[perm]
        if self.config['normalize_latent']:
            latent_backward_reprs = latent_backward_reprs / jnp.linalg.norm(
                latent_backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        
        latents = jnp.where(
            jax.random.uniform(mix_rng, (batch_size, 1)) < self.config['latent_mix_prob'],
            latents,
            latent_backward_reprs,
        )

        return latents

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, latents,
                                            goal_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['forward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['backward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define networks.
        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['fackward_repr_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('forward_repr'),
        )
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('backward_repr'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            activations=getattr(nn, config['activation']),
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, None, ex_actions)),
            backward_repr=(backward_repr_def, (ex_observations, None, ex_actions)),
            target_forward_repr=(copy.deepcopy(forward_repr_def), (ex_observations, None, ex_actions)),
            target_backward_repr=(copy.deepcopy(backward_repr_def), (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, ex_latents, True))
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_forward_repr'] = params['modules_forward_repr']
        params['modules_target_backward_repr'] = params['modules_backward_repr']
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='onestep_fb',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            forward_repr_hidden_dims=(512, 512, 512, 512),  # Forward representation network hidden dimensions.
            backward_repr_hidden_dims=(512, 512, 512, 512),  # Backward representation network hidden dimension.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            fackward_repr_layer_norm=True,  # Whether to use layer normalization for the forward representations.
            backward_repr_layer_norm=True,  # Whether to use layer normalization for the backward representations.
            activation='gelu',  # Activation function.
            latent_dim=128,  # Latent dimension for transition latents.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            normalize_latent=True,  # Whether to normalize backward representations.
            reward_temperature=10,  # Reward weight temperature.
            repr_agg='mean',  # Aggregation method for target forward backward representation.
            q_agg='min',  # Aggregation method for forward-latent inner product.
            orthonorm_coeff=0.0,  # orthonormalization coefficient
            latent_mix_prob=0.5,  # Probability to replace latents sampled from gaussian with backward representations.
            alpha=1.0,  # BC coefficient in reparameterized policy gradient + behavioral cloning (RPG+BC).
            tanh_squash=True,  # Whether to use tanh squash for the actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            num_zero_shot_samples=100_000,  # Number of samples used to infer the zero-shot latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset=ml_collections.ConfigDict(
                dict(
                    dataset_class='Dataset',  # Dataset class name ('GCDataset', 'Dataset', etc.).
                    relabeling=False,  # Whether to relabel rewards.
                    value_p_curgoal=0.2,  # Unused (defined for compatibility with GCDataset).
                    value_p_trajgoal=0.5,  # Unused (defined for compatibility with GCDataset).
                    value_p_randomgoal=0.3,  # Unused (defined for compatibility with GCDataset).
                    value_geom_sample=True,  # Unused (defined for compatibility with GCDataset).
                    actor_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
                    actor_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
                    actor_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
                    actor_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
                    gc_negative=False,  # Unused (defined for compatibility with GCDataset).
                    p_aug=0.0,  # Probability of applying image augmentation.
                    frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
                ),
            ),
        )
    )
    return config
