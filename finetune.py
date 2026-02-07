import json
import importlib
import os
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from agents import agents
from utils.datasets import Dataset, ReplayBuffer
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'experiments', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'exorl-rnd-quadruped-jump', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp_logs', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('zero_shot_restore_path', None, 'Zero-shot agent restore path.')
flags.DEFINE_integer('zero_shot_restore_epoch', None, 'Zero-shot agent restore epoch.')

flags.DEFINE_integer('seed_steps', 10000, 'Number of seed steps.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/td3.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb_run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.enable_wandb:
        setup_wandb(
            wandb_output_dir=FLAGS.save_dir,
            project='onestep-fb', group=FLAGS.wandb_run_group, name=exp_name,
            mode=FLAGS.wandb_mode
        )
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up configs.
    config = FLAGS.agent
    if FLAGS.zero_shot_restore_path is not None:
        with open(os.path.join(FLAGS.zero_shot_restore_path, 'flags.json'), 'r') as f:
            zero_shot_config = ConfigDict(json.load(f)['agent'])
        assert config['latent_dim'] == zero_shot_config['latent_dim']

    # Set up environment and datasets.
    dataset_config = config['dataset']
    dataset_config['discount'] = config['discount']
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=dataset_config['frame_stack'], add_info=True)
    eval_env = make_env_and_datasets(
        FLAGS.env_name, frame_stack=dataset_config['frame_stack'], env_only=True)

    train_dataset = Dataset.create(**train_dataset)
    if val_dataset is not None:
        val_dataset = Dataset.create(**val_dataset)
        zero_shot_dataset_dict = val_dataset
    else:
        zero_shot_dataset_dict = train_dataset

    dataset_module = importlib.import_module('utils.datasets')
    dataset_class = getattr(dataset_module, dataset_config['dataset_class'])
    train_dataset = dataset_class(train_dataset, dataset_config)
    if val_dataset is not None:
        val_dataset = dataset_class(val_dataset, dataset_config)

    # Create example batch and replay buffer.
    example_batch = train_dataset.sample(1)
    
    initial_dataset = dict()
    for k in ['observations', 'actions', 'rewards', 
              'terminals', 'masks', 'next_observations']:
        initial_dataset[k] = train_dataset.dataset[k]
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        initial_dataset, size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    if FLAGS.zero_shot_restore_path is not None:
        # Initialize zero-shot agent.
        zero_shot_agent_class = agents[zero_shot_config['agent_name']]
        zero_shot_agent = zero_shot_agent_class.create(
            FLAGS.seed,
            example_batch,
            zero_shot_config,
        )
        
        # Restore zero-shot agent.
        zero_shot_agent = restore_agent(zero_shot_agent, FLAGS.zero_shot_restore_path, FLAGS.zero_shot_restore_epoch)

        # Infer latent.
        zero_shot_dataset = relabel_dataset(
            FLAGS.env_name, 
            eval_env, 
            zero_shot_dataset_dict, 
        )
        zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dataset), dataset_config)
        zero_shot_batch = zero_shot_dataset.sample(
            zero_shot_config['num_zero_shot_samples'], 
            idxs=np.arange(zero_shot_config['num_zero_shot_samples']),
            relabeling=False,
            augmentation=False
        )
        inferred_latent = zero_shot_agent.infer_latent(zero_shot_batch)
        inferred_latent = np.asarray(inferred_latent)
        example_batch['latent'] = inferred_latent
    else:
        example_batch['latent'] = np.zeros(config['latent_dim'])

    # Initialize agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    if FLAGS.zero_shot_restore_path is not None:
        # Initialize actor.
        zero_shot_actor_params = zero_shot_agent.network.params['modules_actor']
        assert all(jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(
                lambda x, y: x.shape == y.shape, 
                agent.network.params['modules_actor'], 
                zero_shot_actor_params
            ))[0]
        )
        agent.network.params['modules_actor'] = zero_shot_actor_params

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    rng = jax.random.PRNGKey(FLAGS.seed)
    expl_metrics = dict()
    obs, _ = env.reset()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, expl_rng = jax.random.split(rng)

        # Sample transition.
        if i < FLAGS.seed_steps:
            action = agent.sample_actions(observations=obs, temperature=1, seed=expl_rng)
            action = np.asarray(action)

        next_obs, reward, terminated, truncated, info = env.step(action.copy())
        done = terminated or truncated

        replay_buffer.add_transition(
            dict(
                observations=obs,
                actions=action,
                rewards=reward,
                terminals=float(done),
                masks=1.0 - terminated,
                next_observations=next_obs,
            )
        )
        obs = next_obs

        if done:
            expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}
            obs, _ = env.reset()

        if i < FLAGS.seed_steps:
            continue
        elif i == FLAGS.seed_steps:
            assert replay_buffer.size == FLAGS.seed_steps
            
            eval_metrics = {}
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Update agent.
        batch = replay_buffer.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % FLAGS.eval_interval == 0:
            eval_metrics = {}
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            eval_metrics.update({f'evaluation/{k}': v for k, v in eval_info.items()})

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            if FLAGS.enable_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
