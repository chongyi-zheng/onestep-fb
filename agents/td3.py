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


class TD3Agent(flax.struct.PyTreeNode):
    """Twin Delayed Deep Deterministic policy gradient (TD3) agent.
    
    https://arxiv.org/abs/1802.09477
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        masks = batch['masks']
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
        next_qs = self.network.select('target_critic')(next_observations, actions=next_actions)
        if self.config['q_agg'] == 'mean':
            next_q = jnp.mean(next_qs, axis=0)
        else:
            next_q = jnp.min(next_qs, axis=0)
        target_q = rewards + self.config['discount'] * masks * next_q

        # Comppute TD loss.
        qs = self.network.select('critic')(observations, actions=actions, params=grad_params)
        critic_loss = jnp.mean((qs - target_q) ** 2)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }
    
    def actor_loss(self, batch, grad_params):
        """Compute the RPG+BC actor loss."""
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']
        
        # Sample actions.
        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        
        # Compute Q loss.
        qs = self.network.select('critic')(
            observations, actions=q_actions)
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
            'std': action_std.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)

        # Sample latents.
        batch['latents'] = self.sample_latents(batch)

        # Train the critic to predict Q-value.
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Train the actor to maximize the critic.
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        
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
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_latents(self, batch):
        """Sample the fixed latent variables."""
        batch_size = batch['observations'].shape[0]
        latents = jnp.repeat(self.config['latent'][None], batch_size, axis=0)

        return latents

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        if list(observations.shape) == self.config['obs_dims']:
            latents = self.config['latent']
        else:
            latents = jnp.repeat(self.config['latent'][None], observations.shape[0], axis=0)
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

        obs_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['obs_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define value and actor networks.
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
            critic=(critic_def, (ex_observations, None, ex_actions)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_actions)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations, ex_latents, True)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        config['obs_dims'] = obs_dims
        config['latent'] = jnp.asarray(ex_batch['latent'])
    
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='td3',  # Agent name.
            obs_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            latent=ml_collections.config_dict.placeholder(jnp.ndarray),  # Fixed latent variable (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            activation='gelu',  # Activation function.
            latent_dim=128,  # Laplacian latent dimension.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for Q.
            tanh_squash=False,  # Whether to use tanh squash for the actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.2,  # Actor noise clipping threshold.
            alpha=0.0,  # BC coefficient in RPG+BC.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            num_zero_shot_samples=100_000,  # Number of samples used to infer the task-specific latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset=ml_collections.ConfigDict(
                dict(
                    dataset_class='GCDataset',  # Dataset class name ('GCDataset', 'Dataset', etc.).
                    relabeling=False,  # Unused (defined for compatibility with GCDataset).
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
