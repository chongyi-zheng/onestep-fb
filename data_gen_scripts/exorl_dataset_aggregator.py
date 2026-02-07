import glob
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import multiprocessing as mp

from envs import exorl_utils


def load_episode(fn):
    """Load episode from npz files."""
    with open(fn, 'rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relabel_episode(envs, episode):
    """Relabel transitions with task specific rewards.
    
    Note that we relabel the rewards for every task that will be used for RL pre-training and zero-shot adaptation.
    """
    episode['reward'] = dict()
    for task_name, env in envs.items():
        rewards = []
        reward_spec = env.reward_spec()
        states = episode['physics']
        for i in range(states.shape[0]):
            with env.physics.reset_context():
                env.physics.set_state(states[i])
            reward = env.task.get_reward(env.physics)
            reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
            rewards.append(reward)
        rewards = np.array(rewards, dtype=reward_spec.dtype)
        
        episode['reward'][task_name] = rewards
    return episode


class OfflineDatasetAggregator:
    def __init__(self, domain_name, dataset_dir, skip_size, max_size, num_workers, 
                 relabel_reward=True):
        self._domain_name = domain_name
        self._dataset_dir = dataset_dir
        self._skip_size = skip_size
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._relabel_reward = relabel_reward

    def _worker_init_fn(self):
        worker_id = mp.current_process()._identity[0]

        seed = int(np.random.get_state()[1][0] + worker_id)
        np.random.seed(seed)
        random.seed(seed)

    def _worker_fn(self, args):
        (eps_fns, tasks, relabel) = args
        
        envs = dict()
        for task in tasks:
            envs['_'.join(task.split('_')[1:])] = exorl_utils.make_env(task)

        dataset = defaultdict(list)
        if relabel:
            dataset['rewards'] = defaultdict(list)
        for eps_fn in eps_fns:
            episode = load_episode(eps_fn)
            if relabel:
                episode = relabel_episode(envs, episode)

            for k in ['observation', 'action', 'physics']:
                if k in ['observation']:
                    episode_v = episode[k]
                else:
                    episode_v = np.concatenate([episode[k][1:], episode[k][0][None]])
                dataset_k = k if k == 'physics' else k + 's'
                dataset[dataset_k].append(episode_v)

            if relabel:
                for k, v in episode['reward'].items():
                    task_rewards = np.concatenate([v[1:], v[0][None]])
                    dataset['rewards'][k].append(task_rewards)

            resets = np.zeros_like(episode['discount'].squeeze(), dtype=bool)
            resets[0] = True
            dataset['resets'].append(resets)

        for k, v in dataset.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    dataset[k][k_] = np.concatenate(v_)
            else:
                dataset[k] = np.concatenate(v)

        return dataset

    def load(self):
        all_eps_fns = sorted(glob.glob(osp.join(self._dataset_dir, '*.npz')))
        assert len(all_eps_fns) > 0, "Check the exploratory datasets."
        size, eps_fns = 0, []
        for eps_fn in all_eps_fns:
            eps_len = int(eps_fn.split('/')[-1].split('_')[-1].replace('.npz', ''))
            size += eps_len
            if size <= self._skip_size:
                continue
            elif size > (self._skip_size + self._max_size):
                break
            eps_fns.append(eps_fn)

        split_size = int(np.ceil(len(eps_fns) / self._num_workers))
        worker_args = [(eps_fns[i:i + split_size], 
                        [self._domain_name + '_' + task 
                         for task in exorl_utils.ALL_TASKS[self._domain_name]], 
                       self._relabel_reward) 
                       for i in range(0, len(eps_fns), split_size)]
        with mp.Pool(processes=self._num_workers, initializer=self._worker_init_fn) as pool:
            results = pool.map(self._worker_fn, worker_args)

        # aggregate datasets
        dataset = defaultdict(list)
        if self._relabel_reward:
            dataset['rewards'] = defaultdict(list)
        for result in results:
            for k, v in result.items():
                if k == 'rewards':
                    for k_, v_ in v.items():
                        dataset[k][k_].append(v_.squeeze())
                else:
                    dataset[k].append(v.squeeze())

        for k, v in dataset.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    dataset[k][k_] = np.concatenate(v_)
            else:
                dataset[k] = np.concatenate(v)
                
        dataset_size = len(dataset['observations'])
        for k, v in dataset.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    assert len(v_) == dataset_size
            else:
                assert len(v) == dataset_size

        return dataset


def make_dataset(domain_name, dataset_dir, skip_size, max_size, num_workers):
    """Create dataset by using the offline dataset aggregator."""
    aggregator = OfflineDatasetAggregator(domain_name, dataset_dir, skip_size, max_size, num_workers)
    print('Loading and relabeling data...')
    dataset = aggregator.load()
    print('Dataset loaded.')

    return dataset
