import h5py


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
