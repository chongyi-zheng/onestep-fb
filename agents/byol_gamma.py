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


class BYOLGammaAgent(flax.struct.PyTreeNode):
    """Bootstrap your own latent with a geometric goal distribution (BYOL-$\gamma$) agent.
    
    https://arxiv.org/abs/2506.10137
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def repr_loss(self, batch, grad_params):
        """Compute the representation loss."""
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['value_goals']
        
        # Consistency loss.
        phis = self.network.select('obs_repr')(observations, params=grad_params)
        forward_phis = self.network.select('transition')(phis, actions=actions, params=grad_params)
        goal_phis = self.network.select('target_obs_repr')(goals)
        phi_loss = jnp.mean((forward_phis - goal_phis) ** 2)
        
        # Orthonormalization regularization.
        ortho_diag_loss = -(jnp.sum(phis * phis, axis=-1) ** 2)
        ortho_diag_loss = jnp.mean(ortho_diag_loss)
        ortho_off_diag_loss = jnp.mean(jnp.sum(phis * goal_phis, axis=-1) ** 2)
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss
        repr_loss = phi_loss + self.config['orthonorm_coeff'] * ortho_loss

        return repr_loss, {
            'repr_loss': repr_loss,
            'phi_loss': phi_loss,
            'ortho_loss': ortho_loss,
            'ortho_diag_loss': ortho_diag_loss,
            'ortho_off_diag_loss': ortho_off_diag_loss,
            'phi': phis.mean(),
            'goal_phi': goal_phis.mean(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        rewards = batch['latent_rewards']
        latents = batch['latents']
        
        # Sample next actions.
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('target_actor')(next_observations, latents, goal_encoded=True)
        next_actions = next_dist.mode()
        noise = jnp.clip(
            (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # Compute target Q.
        next_qs = self.network.select('target_critic')(
            next_observations, latents, actions=next_actions, goal_encoded=True)
        if self.config['q_agg'] == 'mean':
            next_q = jnp.mean(next_qs, axis=0)
        else:
            next_q = jnp.min(next_qs, axis=0)
        target_q = rewards + self.config['discount'] * next_q

        # Comppute TD loss.
        qs = self.network.select('critic')(
            observations, latents, actions=actions, goal_encoded=True, params=grad_params)
        critic_loss = jnp.mean((qs - target_q) ** 2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }
    
    def actor_loss(self, batch, grad_params):
        """Compute the TD3 actor loss."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']
        
        # Sample actions.
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        
        # Compute Q loss.
        qs = self.network.select('critic')(
            observations, latents, actions=q_actions, goal_encoded=True)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        
        # Compute BC loss.
        bc_loss = jnp.mean((q_actions - actions) ** 2)
        
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
            'std': action_std.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, latent_rng, critic_rng = jax.random.split(rng, 3)

        # Sample latents and intrinsic rewards.
        batch['latents'], batch['latent_rewards'] = self.sample_latents(batch, latent_rng)

        # Train the BYOL-$\gamma$ representations.
        repr_loss, repr_info = self.repr_loss(batch, grad_params)
        for k, v in repr_info.items():
            info[f'repr/{k}'] = v

        # Train the critic using intrinsic rewards defined by the BYOL-$\gamma$ representations.
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Train the actor to maximize the critic.
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = repr_loss + critic_loss + actor_loss
        
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
        self.target_update(new_network, 'obs_repr')
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        """Infer the latent variable using rewards on downstream tasks."""
        phis = self.network.select('obs_repr')(batch['observations'])
        latent = jnp.linalg.lstsq(phis, batch['rewards'])[0]
        if self.config['normalize_latent']:
            latent = latent / jnp.linalg.norm(
                latent, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Sample latent variables and intrisic rewards."""
        batch_size = batch['observations'].shape[0]
        latents = jax.random.normal(rng, shape=(batch_size, self.config['latent_dim']),
                                    dtype=batch['actions'].dtype)
        if self.config['normalize_latent']:
            latents = latents / jnp.linalg.norm(
                latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        # intrinsic rewards defined by the successor feature
        phis = self.network.select('obs_repr')(batch['observations'])
        rewards = (phis * latents).sum(axis=-1)

        return latents, rewards

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, temperature=temperature)
        actions = dist.mode()
        noise = jnp.clip(
            (jax.random.normal(seed, actions.shape) * self.config['actor_noise'] * temperature),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        actions = jnp.clip(actions + noise, -1, 1)

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
        ex_latents = jnp.ones((ex_actions.shape[0], config['latent_dim']), dtype=ex_actions.dtype)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['obs_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define representation, transition, value, and actor networks.
        obs_repr_def = GCValue(
            hidden_dims=config['repr_hidden_dims'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['repr_layer_norm'],
            value_dim=config['latent_dim'],
            num_ensembles=1,
            gc_encoder=encoders.get('obs_repr'),
        )
        transition_def = GCValue(
            hidden_dims=config['repr_hidden_dims'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['repr_layer_norm'],
            value_dim=config['latent_dim'],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('critic'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            activations=getattr(nn, config['activation']),
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            layer_norm=config['actor_layer_norm'],
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            obs_repr=(obs_repr_def, (ex_observations, )),
            transition=(transition_def, (ex_latents, None, ex_actions)),
            critic=(critic_def, (ex_observations, ex_latents, ex_actions, None, True)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            target_obs_repr=(copy.deepcopy(obs_repr_def), (ex_observations, )),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_latents, ex_actions, None, True)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_latents, True)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_obs_repr'] = params['modules_obs_repr']
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='byol_gamma',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            repr_hidden_dims=(512, 512, 512, 512),  # Representation network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            repr_layer_norm=False,  # Whether to use layer normalization for the representation.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            activation='gelu',  # Activation function.
            latent_dim=128,  # Laplacian latent dimension.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            orthonorm_coeff=0.0,  # Orthonormalization regularization coefficient.
            normalize_latent=True,  # Whether to normalize backward representations.
            q_agg='mean',  # Aggregation method for Q.
            alpha=3.0,  # BC coefficient in RPG+BC.
            tanh_squash=True,  # Whether to use tanh squash for the actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.2,  # Actor noise clipping threshold.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            num_zero_shot_samples=100_000,  # Number of samples used to infer the task-specific latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset=ml_collections.ConfigDict(
                dict(
                    dataset_class='GCDataset',  # Dataset class name ('GCDataset', etc.).
                    relabeling=False,  # Whether to relabel rewards.
                    value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
                    value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
                    value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
                    value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
                    actor_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
                    actor_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
                    actor_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
                    actor_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
                    gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
                    p_aug=0.0,  # Probability of applying image augmentation.
                    frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
                ),
            ),
        )
    )
    return config
