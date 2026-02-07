import json
import importlib
import os
import random
import time
from collections import defaultdict

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.exorl_utils import ALL_TASKS as EXORL_ALL_TASKS
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_integer('enable_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('wandb_run_group', 'experiments', 'Run group.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'ogbench-antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp_logs', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/onestep_fb.py', lock_config=False)


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
        
    config = FLAGS.agent

    # Set up environment and dataset.
    dataset_config = config['dataset']
    dataset_config['discount'] = config['discount']
    eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=dataset_config['frame_stack'], add_info=True)

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

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        
        if config['agent_name'] in ['icvf']:
            intention_batch = train_dataset.sample(config['batch_size'])
            batch.update(
                intention_rewards=intention_batch['rewards'],
                intention_masks=intention_batch['masks'],
                intention_goals=intention_batch['value_goals'],
            )
            agent, update_info = agent.update(batch)
        else:
            agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'], augmentation=False)
                if config['agent_name'] in ['icvf']:
                    val_intention_batch = val_dataset.sample(config['batch_size'], augmentation=False)
                    val_batch.update(
                        intention_rewards=val_intention_batch['rewards'],
                        intention_masks=val_intention_batch['masks'],
                        intention_goals=val_intention_batch['value_goals'],
                    )
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.enable_wandb:
                wandb.log(train_metrics, step=i)

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)

            if 'exorl' in FLAGS.env_name:
                domain_name = FLAGS.env_name.split('-')[-1]
                task_infos = EXORL_ALL_TASKS.get(domain_name, None)
            else:
                task_infos = eval_env.unwrapped.task_infos if hasattr(eval_env.unwrapped, 'task_infos') else eval_env.task_infos

            num_tasks = len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                if 'exorl' in FLAGS.env_name:
                    task_name = task_infos[task_id - 1]
                    env_name = '-'.join([FLAGS.env_name, task_name])
                    eval_env = make_env_and_datasets(
                        env_name, frame_stack=dataset_config['frame_stack'], env_only=True)
                else:
                    env_name = FLAGS.env_name
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_env.reset(options=dict(task_id=task_id))

                zero_shot_dataset = relabel_dataset(
                    env_name, 
                    eval_env, 
                    zero_shot_dataset_dict, 
                )
                zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dataset), dataset_config)

                assert zero_shot_dataset.size >= config['num_zero_shot_samples']
                zero_shot_batch = zero_shot_dataset.sample(config['num_zero_shot_samples'], 
                                                           idxs=np.arange(config['num_zero_shot_samples']),
                                                           relabeling=False,
                                                           augmentation=False)
                inferred_latent = agent.infer_latent(zero_shot_batch)
                inferred_latent = np.asarray(inferred_latent)

                eval_info, _, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    task_id=task_id,
                    inferred_latent=inferred_latent,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                if 'exorl' in FLAGS.env_name:
                    metric_names = ['episode.return']
                else:
                    metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
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
