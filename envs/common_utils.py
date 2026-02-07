import h5py


def get_h5_keys(h5file):
    """
    reference: https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/offline_env.py#L20-L28
    """
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_h5_to_dict(file):
    out = {}
    for key, item in file.items():
        if isinstance(item, h5py.Dataset):
            # Load dataset data
            out[key] = item[()]
        elif isinstance(item, h5py.Group):
            # For nested loading, you would need a recursive load function.
            out[key] = load_h5_to_dict(item)
    return out
