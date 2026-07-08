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


class TDJEPAAgent(flax.struct.PyTreeNode):
    """Temporal-Difference Joint-Embedding Predictive Architecture (TD-JEPA) agent.

    https://arxiv.org/abs/2510.00739.
    """

    rng: Any
    network: Any
    z_cov: Any
    config: Any = nonpytree_field()

    def repr_loss(self, batch, grad_params, rng):
        """Compute the symmetric TD-JEPA representation loss."""
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        latents = batch['latents']

        target_phi_next = self.network.select('target_phi')(next_observations)
        target_psi_next = self.network.select('target_psi')(next_observations)

        next_dist = self.network.select('actor')(
            next_observations, latents, goal_encoded=True)
        if self.config['const_std']:
            next_actions = jnp.clip(next_dist.mode(), -1, 1)
        else:
            next_actions = jnp.clip(next_dist.sample(seed=rng), -1, 1)

        target_phi_preds = self.network.select('target_forward_predictor')(
            target_phi_next, latents, actions=next_actions, goal_encoded=True)
        if self.config['repr_agg'] == 'mean':
            target_phi_preds = jnp.mean(target_phi_preds, axis=0)
        else:
            target_phi_preds = jnp.min(target_phi_preds, axis=0)
        forward_target = target_psi_next + self.config['discount'] * target_phi_preds

        target_psi_preds = self.network.select('target_backward_predictor')(
            target_psi_next, latents, actions=next_actions, goal_encoded=True)
        if self.config['repr_agg'] == 'mean':
            target_psi_preds = jnp.mean(target_psi_preds, axis=0)
        else:
            target_psi_preds = jnp.min(target_psi_preds, axis=0)
        backward_target = target_phi_next + self.config['discount'] * target_psi_preds

        phis = self.network.select('phi')(observations, params=grad_params)
        phi_preds = self.network.select('forward_predictor')(
            phis, latents, actions=actions, goal_encoded=True, params=grad_params)
        forward_loss = jnp.mean((phi_preds - forward_target[None]) ** 2)

        psis = self.network.select('psi')(observations, params=grad_params)
        psi_preds = self.network.select('backward_predictor')(
            psis, latents, actions=actions, goal_encoded=True, params=grad_params)
        backward_loss = jnp.mean((psi_preds - backward_target[None]) ** 2)

        # Orthonormality regularization on phi and psi: -E[phi^T phi] + E_{i!=j}[(phi_i^T phi_j)^2].
        batch_size = phis.shape[0]
        I = jnp.eye(batch_size)

        phi_cov = jnp.matmul(phis, phis.T)
        phi_diag_loss = -jnp.diag(phi_cov).mean()
        phi_off_diag_loss = jnp.sum((phi_cov * (1 - I)) ** 2, axis=-1) / (batch_size - 1)
        phi_off_diag_loss = 0.5 * jnp.mean(phi_off_diag_loss)
        phi_ortho_loss = phi_diag_loss + phi_off_diag_loss

        psi_cov = jnp.matmul(psis, psis.T)
        psi_diag_loss = -jnp.diag(psi_cov).mean()
        psi_off_diag_loss = jnp.sum((psi_cov * (1 - I)) ** 2, axis=-1) / (batch_size - 1)
        psi_off_diag_loss = 0.5 * jnp.mean(psi_off_diag_loss)
        psi_ortho_loss = psi_diag_loss + psi_off_diag_loss

        ortho_loss = phi_ortho_loss + psi_ortho_loss
        repr_loss = forward_loss + backward_loss + self.config['orthonorm_coeff'] * ortho_loss

        return repr_loss, {
            'repr_loss': repr_loss,
            'forward_loss': forward_loss,
            'backward_loss': backward_loss,
            'ortho_loss': ortho_loss,
            'phi_ortho_loss': phi_ortho_loss,
            'psi_ortho_loss': psi_ortho_loss,
            'phi_diag_loss': phi_diag_loss,
            'phi_off_diag_loss': phi_off_diag_loss,
            'psi_diag_loss': psi_diag_loss,
            'psi_off_diag_loss': psi_off_diag_loss,
            'phi_mean': phis.mean(),
            'psi_mean': psis.mean(),
            'phi_pred_mean': phi_preds.mean(),
            'psi_pred_mean': psi_preds.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss: maximize T_phi(phi(s), a_hat, z)^T z."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']

        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        phis = self.network.select('phi')(observations)
        phi_preds = self.network.select('forward_predictor')(
            phis, latents, actions=q_actions, goal_encoded=True)
        qs = jnp.einsum('ebd,bd->eb', phi_preds, latents)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
        log_prob = dist.log_prob(actions)

        bc_loss = -log_prob.mean()

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
        rng, latent_rng, repr_rng, actor_rng = jax.random.split(rng, 4)

        # Sample latents.
        batch['latents'] = self.sample_latents(batch, latent_rng)

        # Train the TD-JEPA representations.
        repr_loss, repr_info = self.repr_loss(batch, grad_params, repr_rng)
        for k, v in repr_info.items():
            info[f'repr/{k}'] = v

        # Train the actor to maximize the forward predictor inner products.
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = repr_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
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
        self.target_update(new_network, 'phi')
        self.target_update(new_network, 'psi')
        self.target_update(new_network, 'forward_predictor')
        self.target_update(new_network, 'backward_predictor')

        if self.config['latent_mix_covar']:
            psis = new_network.select('psi')(batch['observations'])
            batch_z_cov = jnp.matmul(psis.T, psis) / psis.shape[0]
            new_z_cov = (
                (1.0 - self.config['tau']) * self.z_cov
                + self.config['tau'] * batch_z_cov
            )
        else:
            new_z_cov = self.z_cov

        return self.replace(network=new_network, rng=new_rng, z_cov=new_z_cov), info

    @jax.jit
    def infer_latent(self, batch):
        """Infer a task latent z from reward-labeled observations via closed-form least squares."""
        observations = batch['observations']
        rewards = batch['rewards']

        psis = self.network.select('psi')(observations)
        latent = jnp.linalg.lstsq(psis, rewards)[0]
        if self.config['normalize_latent']:
            latent = latent / jnp.linalg.norm(
                latent, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        """Sample latents during training: mix uniform-on-sphere with psi(random_state)."""
        rng, latent_rng, perm_rng, mix_rng = jax.random.split(rng, 4)

        goals = batch['goals']
        batch_size = batch['observations'].shape[0]

        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        if self.config['normalize_latent']:
            latents = latents / jnp.linalg.norm(
                latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        perm = jax.random.permutation(perm_rng, jnp.arange(batch_size))
        psi_goals = self.network.select('psi')(goals)
        psi_goals = psi_goals[perm]
        if self.config['latent_mix_covar']:
            inv_cov = jnp.linalg.inv(
                self.z_cov + 1e-6 * jnp.eye(self.config['latent_dim']))
            psi_goals = jnp.matmul(psi_goals, inv_cov)
        if self.config['normalize_latent']:
            psi_goals = psi_goals / jnp.linalg.norm(
                psi_goals, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])

        latents = jnp.where(
            jax.random.uniform(mix_rng, (batch_size, 1)) < self.config['latent_mix_prob'],
            latents,
            psi_goals,
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
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, temperature=temperature)
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
        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        ex_phis = jnp.ones((*ex_actions.shape[:-1], config['phi_dim']))
        ex_psis = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['phi'] = GCEncoder(state_encoder=encoder_module())
            encoders['psi'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        phi_def = GCValue(
            hidden_dims=config['repr_hidden_dims'],
            value_dim=config['phi_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['repr_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('phi'),
        )
        psi_def = GCValue(
            hidden_dims=config['repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['repr_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('psi'),
        )
        forward_predictor_def = GCValue(
            hidden_dims=config['predictor_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['predictor_layer_norm'],
            num_ensembles=2,
        )
        backward_predictor_def = GCValue(
            hidden_dims=config['predictor_hidden_dims'],
            value_dim=config['phi_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['predictor_layer_norm'],
            num_ensembles=2,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            tanh_squash=config['tanh_squash'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            final_fc_init_scale=config['actor_fc_scale'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            phi=(phi_def, (ex_observations,)),
            psi=(psi_def, (ex_observations,)),
            forward_predictor=(forward_predictor_def, (ex_phis, ex_latents, ex_actions, None, True)),
            backward_predictor=(backward_predictor_def, (ex_psis, ex_latents, ex_actions, None, True)),
            target_phi=(copy.deepcopy(phi_def), (ex_observations,)),
            target_psi=(copy.deepcopy(psi_def), (ex_observations,)),
            target_forward_predictor=(
                copy.deepcopy(forward_predictor_def),
                (ex_phis, ex_latents, ex_actions, None, True),
            ),
            target_backward_predictor=(
                copy.deepcopy(backward_predictor_def),
                (ex_psis, ex_latents, ex_actions, None, True),
            ),
            actor=(actor_def, (ex_observations, ex_latents, True)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['grad_norm_clipping'] > 0.0:
            network_tx = optax.chain(
                optax.clip_by_global_norm(config['grad_norm_clipping']),
                optax.adam(learning_rate=config['lr']),
            )
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_phi'] = params['modules_phi']
        params['modules_target_psi'] = params['modules_psi']
        params['modules_target_forward_predictor'] = params['modules_forward_predictor']
        params['modules_target_backward_predictor'] = params['modules_backward_predictor']

        z_cov = jnp.eye(config['latent_dim'])

        return cls(rng, network=network, z_cov=z_cov, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='td_jepa',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            repr_hidden_dims=(512, 512, 512, 512),  # phi/psi encoder hidden dimensions.
            predictor_hidden_dims=(512, 512, 512, 512),  # T_phi/T_psi predictor hidden dimensions.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            repr_layer_norm=True,  # Whether to use layer normalization for the phi/psi encoders.
            predictor_layer_norm=True,  # Whether to use layer normalization for the predictors.
            activation='gelu',  # Activation function.
            phi_dim=256,  # State-encoder (phi) latent dimension.
            latent_dim=50,  # Task-encoder (psi) latent dimension; also the dimension of z.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            normalize_latent=True,  # Whether to normalize sampled/inferred latents to ||z||=sqrt(latent_dim).
            repr_agg='mean',  # Aggregation method for the predictor ensemble in targets and the actor loss.
            q_agg='min',  # Aggregation method for forward-latent inner product.
            orthonorm_coeff=1.0,  # Orthonormalization coefficient for phi and psi.
            latent_mix_prob=0.5,  # Probability of using a uniform-on-sphere latent vs psi(random_state).
            latent_mix_covar=False,  # Whether to covariance-correct psi-goal latents before mixing.
            alpha=0.0,  # BC coefficient (log-prob style).
            tanh_squash=True,  # Whether to use tanh squash for the actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            grad_norm_clipping=-1.0,  # Global gradient norm clipping bound (negative => no clipping).
            num_zero_shot_samples=100_000,  # Number of samples used to infer the zero-shot latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name ('mlp', 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset=ml_collections.ConfigDict(
                dict(
                    dataset_class='GCDataset',
                    relabeling=False,
                    value_p_curgoal=0.2,
                    value_p_trajgoal=0.5,
                    value_p_randomgoal=0.3,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=1.0,
                    actor_p_randomgoal=0.0,
                    actor_geom_sample=False,
                    gc_negative=False,
                    p_aug=0.0,
                    frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
                ),
            ),
        )
    )
    return config
