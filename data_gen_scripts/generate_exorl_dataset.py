import os
import os.path as osp

import h5py
import numpy as np
from absl import app, flags

from exorl_dataset_aggregator import make_dataset


FLAGS = flags.FLAGS

flags.DEFINE_string('domain_name', 'walker', 'Domain name.')
flags.DEFINE_string('expl_agent_name', 'rnd', 'Exploratory agent name.')
flags.DEFINE_string('dataset_dir', '~/.exorl/expl_datasets', 'Download the dataset to this directory.')
flags.DEFINE_string('save_path', None, 'Save the dataset to this path.')
flags.DEFINE_integer('num_workers', 16, 'Number of workers to collect transitions.')
flags.DEFINE_integer('skip_size', 0, 'Number of transitions to skip in the dataset.')
flags.DEFINE_integer('dataset_size', 5_000_000, 'Size of the dataset.')


def save_dict_to_h5(file, dictionary, path='/'):
    """Save dataset dict to a HDF5 file."""
    for key, item in dictionary.items():
        # Keys must be strings in HDF5
        if not isinstance(key, str):
            key = str(key)
        
        # Determine if the item is a dictionary or data
        if isinstance(item, dict):
            # Create a new group for nested dictionaries
            group_path = f'{path}{key}/'
            save_dict_to_h5(file, item, path=group_path)
        elif isinstance(item, (np.ndarray, np.generic, list, str, bytes)):
            # Save data types as datasets
            file[f"{path}{key}"] = item
        else:
            raise ValueError(f'Cannot save {type(item)} type')


def main(_):
    dataset_dir = osp.expanduser(FLAGS.dataset_dir)
    dataset_dir = osp.join(dataset_dir, FLAGS.domain_name, FLAGS.expl_agent_name, 'buffer')
    print(f"dataset dir: {dataset_dir}")

    # Create dataset.
    dataset = make_dataset(
        FLAGS.domain_name,
        dataset_dir,
        FLAGS.skip_size, 
        FLAGS.dataset_size,
        FLAGS.num_workers,
    )

    # Save dataset.
    save_path = osp.expanduser(FLAGS.save_path)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        save_dict_to_h5(f, dataset)

    print("Save dataset to: {}".format(save_path))


if __name__ == '__main__':
    app.run(main)
