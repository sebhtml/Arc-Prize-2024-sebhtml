import numpy as np
import h5py

class SampleInputTokens:
    def __init__(self, input_state: str, full_move_counter: str, current_state: str, action: str):
        self._input_state = input_state
        self._full_move_counter = full_move_counter
        self._current_state = current_state
        self._action = action


def __get_np_structured_array_dtype():
    """
    See https://numpy.org/doc/stable/user/basics.rec.html
    """
    composite_dtype = [('input_state', 'uint8', (60)), ('full_move_counter', 'uint8', (7)), (
        'current_state', 'uint8', (60)), ('action', 'uint8', (60)), ('action_value', 'float32')]
    return composite_dtype


def create_file_storage(h5_file_path):
    f = h5py.File(h5_file_path, "w")
    sa_dtype = __get_np_structured_array_dtype()
    _dataset = f.create_dataset(
        "samples", shape=(0,), dtype=np.dtype(sa_dtype), maxshape=(None,))
    return f


def __to_h5_sample(sample: tuple[SampleInputTokens, float]):
    input_state: str = sample[0]._input_state
    full_move_counter: str = sample[0]._full_move_counter
    current_state: str = sample[0]._current_state
    action: str = sample[0]._action
    action_value: str = sample[1]
    return (input_state, full_move_counter, current_state, action, action_value)


def append_to_file_storage(h5_file_path: str, train_action_examples):
    # a: Read/write if exists, create otherwise
    size = len(train_action_examples)
    f = h5py.File(h5_file_path, "a")

    # Get datasets
    samples = f["samples"]

    # Size
    current_size = samples.shape[0]

    # Resize
    samples.resize((current_size + size, ))

    # Do the writes
    samples[current_size:] = list(map(__to_h5_sample, train_action_examples))