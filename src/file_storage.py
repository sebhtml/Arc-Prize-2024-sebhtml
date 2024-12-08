import numpy as np
import h5py
from itertools import tee
from context import ExampleInputTokens, StateActionExample


def __get_np_structured_array_dtype():
    """
    See https://numpy.org/doc/stable/user/basics.rec.html
    """
    composite_dtype = [('attended_example_input', 'uint8', (60)), (
        'attended_current_state', 'uint8', (60)), ('attended_action', 'uint8', (60)), ('action_value', 'float32')]
    return composite_dtype


def create_file_storage(h5_file_path):
    # We disable HDF5 locking because it does not work properly on runpod.io MooseFS file system.
    f = h5py.File(h5_file_path, "w", locking=False)
    sa_dtype = __get_np_structured_array_dtype()
    _dataset = f.create_dataset(
        "examples", shape=(0,), dtype=np.dtype(sa_dtype), maxshape=(None,))
    return f


def __to_h5_example(example: tuple[ExampleInputTokens, float]):
    attended_example_input = example[0].attended_example_input()
    attended_current_state = example[0].attended_current_state()
    attended_action = example[0].attended_action()
    action_value = example[1]
    return (attended_example_input, attended_current_state, attended_action, action_value)


def append_to_file_storage(f: h5py.File, train_action_examples):
    size = len(train_action_examples)

    # Get datasets
    examples = f["examples"]

    # Size
    current_size = examples.shape[0]

    # Resize
    examples.resize((current_size + size, ))

    # Do the writes
    examples[current_size:] = list(map(__to_h5_example, train_action_examples))


class FileStorageReader:
    def __init__(self, h5_file_path):
        self.f = h5py.File(h5_file_path, "r")

    def size(self):
        examples = self.f["examples"]
        return examples.shape[0]

    def get_action_value_min_max(self):

        examples = self.f["examples"]
        accumulator_min = examples[0]['action_value']
        accumulator_max = accumulator_min

        size = self.size()
        block_size = 4096
        idx = 0

        while idx < size:
            upper = min(idx + block_size, size)
            block_examples = examples[idx:upper]
            block_action_values = map(
                lambda example: example['action_value'], block_examples)

            min_it, max_it = tee(block_action_values)

            # Min
            accumulator_min = min(accumulator_min, min(min_it))

            # Max
            accumulator_max = max(accumulator_max, max(max_it))

            idx += block_size

        return accumulator_min, accumulator_max

    def get(self, idx) -> StateActionExample:
        examples = self.f["examples"]
        example = examples[idx]
        attended_example_input = example['attended_example_input'].tolist()
        attended_current_state = example['attended_current_state'].tolist()
        attended_action = example['attended_action'].tolist()
        action_value = example['action_value']
        tokens = ExampleInputTokens(
            attended_example_input, attended_current_state, attended_action)
        return StateActionExample(tokens, action_value)


class FileStorageWriter:
    def __init__(self, h5_file_path):
        create_file_storage(h5_file_path)
        # a: Read/write if exists, create otherwise
        # We disable HDF5 locking because it does not work properly on runpod.io MooseFS file system.
        self.f = h5py.File(h5_file_path, "a", locking=False)

    def append(self, train_action_examples):
        append_to_file_storage(self.f, train_action_examples)
