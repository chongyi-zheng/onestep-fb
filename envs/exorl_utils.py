import dataclasses
from collections import OrderedDict, deque
import typing as tp
from typing import Any
import os
import h5py

from dm_env import Environment
from dm_env import StepType, specs
import numpy as np

from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels

import envs.custom_dmc_tasks as cdmc
from envs.dmc2gymnasium import DMControlToGymnasiumWrapper
from envs.common_utils import load_h5_to_dict


DEFAULT_DATASET_DIR = '~/.exorl/data'
INFO_KEYS_WITH_DTYPES = dict(
    physics=np.float64,
)
ALL_TASKS = dict(
    walker=['flip', 'run', 'stand', 'walk'],
    cheetah=['run', 'run_backward', 'walk', 'walk_backward'],
    quadruped=['jump', 'run', 'stand', 'walk'],
    point_mass_maze=['reach_bottom_left', 'reach_bottom_right',
                     'reach_top_left', 'reach_top_right'],
    jaco=['reach_bottom_left', 'reach_bottom_right',
          'reach_top_left', 'reach_top_right'],
)

S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper(Environment):
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)

        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenJacoObservationWrapper(EnvWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for spec in wrapped_obs_spec.values():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((int(np.prod(spec.shape))  # type: ignore
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def observation_spec(self) -> tp.Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')  # type: ignore
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():  # type: ignore
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)


class ActionRepeatWrapper(EnvWrapper):
    def __init__(self, env: tp.Any, num_repeats: int) -> None:
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedTimeStepWrapper(EnvWrapper):
    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


def _make_jaco(obs_type, domain, task, action_repeat, seed, image_wh=64) -> FlattenJacoObservationWrapper:
    del domain

    env = cdmc.make_jaco(task, obs_type, seed, image_wh=image_wh)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, action_repeat, seed, 
              image_wh=64):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=image_wh, width=image_wh, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    return env


def make_env(
    env_name: str, 
    obs_type='states', 
    action_repeat=1, 
    seed=1, 
    image_wh=64,
) -> EnvWrapper:
    assert obs_type in ['states', 'pixels']
    env_name = env_name.replace('-', '_')
    if env_name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = env_name.split('_', 3)
    else:
        domain, task = env_name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)

    make_fn = _make_jaco if domain == 'jaco' else _make_dmc
    env = make_fn(obs_type, domain, task, action_repeat, seed, 
                  image_wh=image_wh)

    if obs_type != 'pixels':
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)

    return env


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    dataset_path=None,
    compact_dataset=False,
    env_only=False,
    dataset_only=False,
    cur_env=None,
    add_info=False,
    **env_kwargs,
):
    """Make ExORL environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save/load the datasets.
        dataset_path: (Optional) Path to the dataset file.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).
        add_info: Whether to add observation information ('maze_layout', 'agent_pos', 'targets_pos', etc.) to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.

    Returns:
        Depending on flags:
        - env_only=True: environment
        - dataset_only=True: (train_dataset, val_dataset)
        - both False: (env, train_dataset, val_dataset)
    """
    # Create environment if needed
    splits = dataset_name.split('-')
    env = cur_env
    assert len(splits) >= 2
    if len(splits) == 2:
        domain_name = dataset_name.split('-')[-1]
        task_name = 'walk' if domain_name != 'jaco' else 'reach_bottom_left'
        env_name = '-'.join([domain_name, task_name])
    else:
        dataset_name = '-'.join(splits[:2])
        domain_name, task_name = splits[1], splits[2]
        env_name = '-'.join(splits[1:])
    if not dataset_only:
        assert domain_name in ALL_TASKS and task_name in ALL_TASKS[domain_name]
        env = make_env(env_name, **env_kwargs)
        env = DMControlToGymnasiumWrapper(env, render_mode='rgb_array')

    if env_only:
        return env

    # Load datasets.
    if dataset_path is None:
        dataset_dir = os.path.expanduser(dataset_dir)
        train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.hdf5')
        val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.hdf5')
    else:
        train_dataset_path = dataset_path
        val_dataset_path = dataset_path.replace('.hdf5', '-val.hdf5')

    ob_dtype = np.float32
    action_dtype = np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=add_info,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=add_info,
    )

    if not add_info:
        # Remove information keys.
        for k in ['physics']:
            if k in train_dataset:
                del train_dataset[k]
            if k in val_dataset:
                del val_dataset[k]

    if dataset_only:
        return train_dataset, val_dataset
    else:
        return env, train_dataset, val_dataset


def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False, add_info=False,
                 info_keys_with_dtypes=INFO_KEYS_WITH_DTYPES):
    """Load ExORL dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        add_info: Whether to add observation information ('physics', etc.) to the dataset.

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
        If `add_info` is True, the dictionary may also contain additional keys for observation information.
    """
    with h5py.File(dataset_path, 'r') as file:
        dataset = load_h5_to_dict(file)
    
    for k, v in dataset.items():
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        elif k not in info_keys_with_dtypes:
            dtype = np.float32
        if isinstance(v, dict):
            for k_, v_ in v.items():
                dataset[k][k_] = v_.astype(dtype)
        else:
            dataset[k] = v.astype(dtype)

    if add_info:
        for k, dtype in info_keys_with_dtypes.items():
            assert k in dataset
            dataset[k] = dataset[k].astype(dtype)
    else:
        for k in info_keys_with_dtypes:
            del dataset[k]

    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                             |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # ------------------------------------------------------------------------
    #            'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    #            'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # (optional) 'rewards'     : [r0, r1, r2, r3, r4, r0, r1, r2, r3, r4, ...]
    #            'resets'      : [ 1,  0,  0,  0,  0,  1,  0,  0,  0,  0, ...]
    
    if compact_dataset:
        # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
        # `next_observations[t]` by using `observations[t + 1]`.
        # Our goal is to have the following structure:
        #                             |<--- traj 1 --->|  |<--- traj 2 --->|  ...
        # ------------------------------------------------------------------------
        #            'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
        #            'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
        # (optional) 'rewards'     : [r0, r1, r2, r3, r4, r0, r1, r2, r3, r4, ...]
        #            'resets'      : [ 1,  0,  0,  0,  0,  1,  0,  0,  0,  0, ...]
        #            'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
        #            'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

        if isinstance(dataset['rewards'], dict):
            for k, v in dataset['rewards'].items():
                dataset['_'.join(['rewards', k])] = dataset['rewards'][k]
            del dataset['rewards']

        new_terminals = np.concatenate([dataset['resets'][1:], [1.0]])
        dataset['valids'] = 1.0 - new_terminals
        dataset['terminals'] = np.minimum(
            new_terminals + np.concatenate([new_terminals[1:], [1.0]]), 
            1.0
        ).astype(np.float32)
    else:
        # Regular dataset: Generate `next_observations` by shifting `observations`.
        # Our goal is to have the following structure:
        #                                  |<- traj 1 ->|  |<- traj 2 ->|  ...
        # ---------------------------------------------------------------------
        #            'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
        #            'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
        # (optional) 'rewards'          : [r0, r1, r2, r3, r0, r1, r2, r3, ...]
        #            'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
        #            'prev_actions'     : [ 0, a0, a1, a2,  0, a0, a1, a2, ...]
        #            'next_actions'     : [a1, a2, a3, a4, a1, a2, a3, a4, ...]
        #            'resets'           : [ 1,  0,  0,  0,  1,  0,  0,  0, ...]
        #            'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]

        new_terminals = np.concatenate([dataset['resets'][1:], [1.0]])
        ob_mask = (1.0 - new_terminals).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        prev_action_mask = np.concatenate([[True], ob_mask[1:], [False]])
        if action_dtype == np.uint8:
            masked_actions = np.concatenate([
                [0],
                np.where(ob_mask, dataset['actions'], 0)
            ])
        elif issubclass(action_dtype, np.floating):
            masked_actions = np.concatenate([
                np.zeros_like(dataset['actions'][:1]),
                np.where(ob_mask.reshape([*dataset['actions'].shape[:-1], 1]), dataset['actions'], 0)
            ])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['prev_actions'] = masked_actions[prev_action_mask]
        dataset['next_actions'] = dataset['actions'][next_ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        if 'rewards' in dataset:
            if isinstance(dataset['rewards'], dict):
                for k, v in dataset['rewards'].items():
                    dataset['_'.join(['rewards', k])] = dataset['rewards'][k][ob_mask]
                del dataset['rewards']
            else:
                dataset['rewards'] = dataset['rewards'][ob_mask]
        dataset['resets'] = dataset['resets'][ob_mask]
        dataset['terminals'] = np.concatenate([dataset['resets'][1:], [1.0]]).astype(np.float32)

        if add_info:
            for k in info_keys_with_dtypes:
                dataset[k] = dataset[k][ob_mask]
    
    dataset_size = dataset['observations'].shape[0]
    for v in dataset.values():
        assert v.shape[0] == dataset_size
    
    return dataset
