import numpy as np
import h5py
from itertools import tee


class SampleInputTokens:
    def __init__(self, input_state: str, current_state: str, action: str):
        self._input_state = input_state
        self._current_state = current_state
        self._action = action


def __get_np_structured_array_dtype():
    """
    See https://numpy.org/doc/stable/user/basics.rec.html
    """
    composite_dtype = [('input_state', 'uint8', (60)), (
        'current_state', 'uint8', (60)), ('action', 'uint8', (60)), ('action_value', 'float32')]
    return composite_dtype


def create_file_storage(h5_file_path):
    # We disable HDF5 locking because it does not work properly on runpod.io MooseFS file system.
    f = h5py.File(h5_file_path, "w", locking=False)
    sa_dtype = __get_np_structured_array_dtype()
    _dataset = f.create_dataset(
        "samples", shape=(0,), dtype=np.dtype(sa_dtype), maxshape=(None,))
    return f


def __to_h5_sample(sample: tuple[SampleInputTokens, float]):
    input_state: str = sample[0]._input_state
    current_state: str = sample[0]._current_state
    action: str = sample[0]._action
    action_value: str = sample[1]
    return (input_state, current_state, action, action_value)


def append_to_file_storage(f: h5py.File, train_action_examples):
    size = len(train_action_examples)

    # Get datasets
    samples = f["samples"]

    # Size
    current_size = samples.shape[0]

    # Resize
    samples.resize((current_size + size, ))

    # Do the writes
    samples[current_size:] = list(map(__to_h5_sample, train_action_examples))


class FileStorageReader:
    def __init__(self, h5_file_path):
        self.f = h5py.File(h5_file_path, "r")

    def size(self):
        samples = self.f["samples"]
        return samples.shape[0]

    def get_action_value_min_max(self):

        samples = self.f["samples"]
        accumulator_min = samples[0]['action_value']
        accumulator_max = accumulator_min

        size = self.size()
        block_size = 4096
        idx = 0

        while idx < size:
            upper = min(idx + block_size, size)
            block_samples = samples[idx:upper]
            block_action_values = map(
                lambda sample: sample['action_value'], block_samples)

            min_it, max_it = tee(block_action_values)

            # Min
            accumulator_min = min(accumulator_min, min(min_it))

            # Max
            accumulator_max = max(accumulator_max, max(max_it))

            idx += block_size

        return accumulator_min, accumulator_max

    def get(self, idx) -> tuple[SampleInputTokens, float]:
        samples = self.f["samples"]
        sample = samples[idx]
        input_state = sample['input_state'].tolist()
        current_state = sample['current_state'].tolist()
        action = sample['action'].tolist()
        action_value = sample['action_value']
        tokens = SampleInputTokens(
            input_state, current_state, action)
        return (tokens, action_value)


class FileStorageWriter:
    def __init__(self, h5_file_path):
        create_file_storage(h5_file_path)
        # a: Read/write if exists, create otherwise
        # We disable HDF5 locking because it does not work properly on runpod.io MooseFS file system.
        self.f = h5py.File(h5_file_path, "a", locking=False)

    def append(self, train_action_examples):
        append_to_file_storage(self.f, train_action_examples)
