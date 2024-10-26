# Author:
# Sebastien Boisvert <sebhtml@protonmail.com>
#
#

# Hardware used:

# Legion
# CPU: AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
# GPU: NVIDIA GeForce RTX 4060 8188MiB
# RAM: MemTotal:       32023592 kB

# Kaggle
# CPU:
# GPU: NVIDIA P100
# RAM:

# Runpod
# - NVIDIA A40 48 GB VRAM
# - NVIDIA RTX A4000 16 GB VRAM

# This software used reinforcement learning.
# It uses Q-learning.
# See https://en.wikipedia.org/wiki/Q-learning

import os  # nopep8
# os.system("pip uninstall fastai torchvision torchaudio")  # nopep8
# For TPUs # nopep8
# os.system(   # nopep8
#    "pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html")   # nopep8
# os.system("pip install xformers")  # nopep8


# import torch_xla
# import torch_xla.core.xla_model as xm
from typing import List
import random
import math
import copy
import numpy as np
import torch
import json
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
import sys
from file_storage import SampleInputTokens, create_file_storage, FileStorageWriter, FileStorageReader
from model import DecoderOnlyTransformerModel

device = torch.device("cuda")

# device = xm.xla_device()
# device = torch.device("cpu")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#
# /kaggle/input/arc-prize-2024/arc-agi_training_challenges.json
# /kaggle/input/arc-prize-2024/arc-agi_training_solutions.json
#
# /kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json
# /kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json
#
# /kaggle/input/arc-prize-2024/arc-agi_test_challenges.json
#
# /kaggle/input/arc-prize-2024/sample_submission.json

# Paths
# On Kaggle
# kaggle_input_path = "/kaggle/input/arc-prize-2024"
# logs_path = "/workspace/logs"
# On runpod
kaggle_input_path = "/workspace/kaggle-input"
logs_path = "/workspace/logs"
models_path = "/workspace/models"
model_file_path = f"{models_path}/2024-09-29-q-network.pth"
selected_puzzle_id = "3aa6fb7a"
train_dataset_path = f"/workspace/train_datasets/{selected_puzzle_id}.hdf5"

# Model configuration
# See https://arcprize.org/play?task=3aa6fb7a
# multiple of 4 for NVIDIA cublas WMMA
context_size = 188
# Each cell has one color and there are 10 colors.
cell_value_size = 10
puzzle_width = 7
puzzle_height = 7
# Hidden size
d_model = 256
# Feed-forward size in transformer block
d_ff = 768
num_classes = 128
shuffle_train_samples = True
# In "Llama 2: Open Foundation and Fine-Tuned Chat Models" https://arxiv.org/abs/2307.09288, they do gradient clipping with norm=1.0
max_grad_norm: float = 1.0
num_heads = 8
dropout = 0.2
num_layers = 4
vocab_size = 128
# For batch_size:
# - 8192 for the TPU machine since "TPU-v3-8" has 330 GB RAM
# - 512 for TPU-v3-8 with 1 TPU
# - 512 for the NVIDIA P100 GPU since "P100" has 16 GB VRAM
# - 1024 for CPU since "CPU" has 29 GB RAM
# On runpod:
# - 1536 with NVIDIA A40 (48 GB VRAM)
# - 512 with NVIDIA A4000 (16 GB VRAM)
batch_size = 1024
lr = 0.0001
# In "Llama 2: Open Foundation and Fine-Tuned Chat Models" https://arxiv.org/abs/2307.09288, they use a weight decay of 0.1
# In "Grandmaster-Level Chess Without Search" https://arxiv.org/html/2402.04494v1, they don't say what weight decay they used.
weight_decay = 0.1
discount = 0.99
num_epochs = 1
# Use 20000 for dev, and use 25088000 for training the model.
total_train_samples = 25088000
padding_char = ' '
generate_train_samples: bool = False
stop_after_generating_samples = False
print_model_outputs: bool = False
load_model: bool = True
train_model: bool = False
# Available modes are:
# - randomize
# - identity
# - zero
input_gen_mode = "identity"
current_gen_mode = "zero"


class QLearningAction:
    def __init__(self, row, col, cell_value):
        self.__row = row
        self.__col = col
        self.__cell_value = cell_value

    def row(self) -> int:
        return self.__row

    def col(self) -> int:
        return self.__col

    def cell_value(self) -> int:
        return self.__cell_value


class Cell:
    def __init__(self, value):
        self.__value = value
        self.__changes = 0

    def value(self) -> int:
        return self.__value

    def changes(self) -> int:
        return self.__changes

    def set_value(self, value):
        self.__value = value
        self.__changes += 1


def input_state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = state[row][col].value()
            output += str(value)
        output += "\n"
    return output


def get_full_move_counter(state) -> str:
    """
    Get the full move counter since the beginning of the playout
    """
    full_move_counter: int = 0
    for row in range(len(state)):
        for col in range(len(state[row])):
            changes = state[row][col].changes()
            if changes > 1:
                raise Exception(
                    f"Current state has a cell with {changes}, which is illegal because it is greater than 1.")
            full_move_counter += changes

    output: str = str(full_move_counter).rjust(2, padding_char)
    output += "\n"
    return output


def current_state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if state[row][col].changes() == 0:
                value = '_'
            else:
                value = state[row][col].value()
            output += str(value)
        output += "\n"
    return output


def action_to_text(state, action: QLearningAction) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if row == action.row() and col == action.col():
                value = action.cell_value()
            else:
                value = '_'
            output += str(value)
        output += "\n"
    return output


def compute_action_token(action: QLearningAction, puzzle_height: int, cell_value_size: int) -> int:
    # action a
    # For example, a puzzle with a grid 7x7 has 7*7*10 possible actions.
    # Points (a, b, c) are in a coordinate system of size (WIDTH, HEIGHT, VALUES)
    action_token = action.col() * puzzle_height * cell_value_size + \
        action.row() * cell_value_size + action.cell_value()
    return action_token


def get_state_texts(input_state, current_state):
    input_state_text = ""
    input_state_text += "ini" + "\n"
    input_state_text += input_state_to_text(input_state)

    full_move_counter = ""
    full_move_counter += "cnt" + "\n"
    full_move_counter += get_full_move_counter(current_state)

    current_state_text = ""
    current_state_text += "cur" + "\n"
    current_state_text += current_state_to_text(current_state)

    return input_state_text, full_move_counter, current_state_text


def tokenize_sample_input(input_state, current_state, action: QLearningAction) -> SampleInputTokens:
    """
    Tokenize a sample input for the Q-network Q(s, a).
    Note that:
    - s contains the state which is (input_state, current_state)
    - a contains the action which is (next_state)
    """

    # state s
    input_state_text, full_move_counter, current_state_text = get_state_texts(
        input_state, current_state)

    action_text = ""
    action_text += "act" + "\n"
    action_text += action_to_text(current_state, action)

    return SampleInputTokens(
        text_to_tokens(input_state_text),
        text_to_tokens(full_move_counter),
        text_to_tokens(current_state_text),
        text_to_tokens(action_text)
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def tokens_to_text(sample_input_tokens: SampleInputTokens) -> str:
    # TODO add a method in SampleInputTokens to get a list of all tokens in a list.
    tokens: List[int] = sample_input_tokens._input_state + sample_input_tokens._full_move_counter + \
        sample_input_tokens._current_state + sample_input_tokens._action
    return "".join(map(chr, tokens))


def reward(expected_cell_value, cell_value) -> int:
    if expected_cell_value == cell_value:
        return 1.0
    else:
        return -1.0


def get_q_star_action_value(state, action: QLearningAction, example_output, discount) -> int:
    """
    - discount is gamma
    - Q*(s, a) = gamma^0 * r_{t+1} + gamma^1* r_{t+1} + gamma^2 * r_{t+2} + ...
    """
    # Immediate reward is not discounted.
    immediate_reward = reward(
        example_output[action.row()][action.col()], action.cell_value())
    # Discounted future rewards
    maximum_sum_of_discounted_future_rewards = 0.0
    t = 1

    for row in range(len(state)):
        for col in range(len(state[row])):
            # Skip cell because it was already counted as the immediate reward.
            if row == action.row() and col == action.col():
                continue
            # A cell can only be changed once.
            if state[row][col].changes() == 1:
                continue
            # Maximize future expected discounted rewards
            future_reward = reward(
                example_output[row][col], example_output[row][col])
            discounted_reward = discount**t * future_reward
            maximum_sum_of_discounted_future_rewards += discounted_reward
            t += 1

    action_value = immediate_reward + maximum_sum_of_discounted_future_rewards
    return action_value


def generate_initial_cell_value(state, row, col, mode) -> int:
    if mode == "randomize":
        return random.randrange(0, cell_value_size)
    elif mode == "identity":
        return state[row][col]
    elif mode == "zero":
        return 0


def get_starting_current_state(state, cell_value_size, mode):
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = generate_initial_cell_value(current_state, row, col, mode)
            current_state[row][col] = Cell(value)
    return current_state


def generate_cell_actions(current_state, cell_value_size) -> list[QLearningAction]:
    candidate_actions = []
    assert current_state != None
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].changes() > 0:
                continue
            for new_value in range(cell_value_size):
                action = QLearningAction(row, col, new_value)
                candidate_actions.append(action)
    np.random.shuffle(candidate_actions)
    return candidate_actions


def generate_action_examples(puzzle_example, cell_value_size):
    (example_input, example_output) = puzzle_example
    action_examples = []
    example_input = get_starting_current_state(
        example_input, cell_value_size, input_gen_mode)
    current_state = get_starting_current_state(
        example_output, cell_value_size, current_gen_mode)

    assert current_state != None

    while current_state != example_output:
        best_next_state = None
        best_action_value = None
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)

        if len(candidate_actions) == 0:
            break

        # print(f"candidate_actions {len(candidate_actions)}")
        for candidate_action in candidate_actions:
            next_state = copy.deepcopy(current_state)
            row = candidate_action.row()
            col = candidate_action.col()
            cell_value = candidate_action.cell_value()
            next_state[row][col].set_value(cell_value)
            input_text = tokenize_sample_input(
                example_input, current_state, candidate_action)
            # Use Q*(s, a) for the action-value.
            action_value = get_q_star_action_value(
                current_state, candidate_action, example_output, discount)
            example = (input_text, action_value)
            action_examples.append(example)
            if best_action_value == None or action_value > best_action_value:
                best_next_state = next_state
                best_action_value = action_value

        assert best_next_state != None
        current_state = best_next_state
        assert current_state != None

    return action_examples


def get_puzzle_solution(venue, puzzle_id):
    solutions_file = f"{kaggle_input_path}/arc-agi_{venue}_solutions.json"
    f = open(solutions_file)
    solutions_data = json.load(f)
    solution = solutions_data[puzzle_id][0]
    return solution


def load_puzzle_examples(venue, puzzle_id, example_type):
    """
    - venue is "training" or "evaluation" or "test"
    - example_type is "train" or "test"
    Note that for the "test" venue, no solutions are provided.
    """
    challenges_file = f"{kaggle_input_path}/arc-agi_{venue}_challenges.json"
    f = open(challenges_file)
    challenges_data = json.load(f)
    puzzle_challenges_data = challenges_data[puzzle_id]
    puzzle_examples = puzzle_challenges_data[example_type]

    puzzle_venue_examples = []
    for puzzle_example in puzzle_examples:
        example_input = puzzle_example["input"]
        example_output = None
        if venue == "test":
            pass
        elif example_type == "train":
            example_output = puzzle_example["output"]
        else:
            example_output = get_puzzle_solution(venue, puzzle_id)

        example = (example_input, example_output)
        puzzle_venue_examples.append(example)
        # TODO don't break.
        break
    return puzzle_venue_examples

# TODO this function should receive just one puzzle example


def generate_train_action_examples(puzzle_examples, cell_value_size):
    """
    Generate (state, action, action_value) experience samples for all puzzle examples.
    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value)
    until the end of the game is reached.
    This is essentially a wrong incorrect half-baked MCTS (Monte Carlo tree search) in the sense that it's a
    tree search. But there is no Monte Carlo or Markov Chain involved here since we are lazy at best.
    """
    train_examples = []
    for puzzle_example in puzzle_examples:
        action_examples = generate_action_examples(
            puzzle_example, cell_value_size)
        train_examples += action_examples
    return train_examples


def make_sample_tensor(sample_input_tokens: SampleInputTokens):
    input_tokens: List[int] = sample_input_tokens._input_state + sample_input_tokens._full_move_counter + \
        sample_input_tokens._current_state + sample_input_tokens._action
    if len(input_tokens) > context_size:
        raise Exception(
            f"text ({len(input_tokens)} tokens) is too large to fit in context ! Increase context_size ({context_size})")
    item_input = [torch.tensor(sample_input_tokens._input_state),
                  torch.tensor(sample_input_tokens._full_move_counter),
                  torch.tensor(sample_input_tokens._current_state),
                  torch.tensor(sample_input_tokens._action)
                  ]
    return item_input


def bin_action_value(action_value: float, minimum_action_value: float, maximum_action_value: float, num_classes: int) -> float:
    """
    convert action_value to { 0, 1, ..., num_classes - 1 }
    Example:
    action_value: 3.0
    _minimum_action_value: -4.0
    _maximum_action_value: 7.0
    action_value_bin = (3.0 - -4.0) / (7.0 - -4.0)
    """
    action_value_bin = math.floor(
        ((action_value - minimum_action_value) / (maximum_action_value - minimum_action_value)) * (num_classes - 1))
    return action_value_bin


class MyDataset(Dataset):
    def __init__(self, h5_file_path):
        self.reader = FileStorageReader(h5_file_path)
        min_value, max_value = self.reader.get_action_value_min_max()
        self._minimum_action_value = min_value
        self._maximum_action_value = max_value
        print(f"_minimum_action_value {self._minimum_action_value}")
        print(f"_maximum_action_value {self._maximum_action_value}")

    def __len__(self):
        return self.reader.size()

    def __getitem__(self, idx):
        example = self.reader.get(idx)

        input_tokens = example[0]
        item_input = make_sample_tensor(input_tokens)

        action_value = example[1]
        action_value_bin = bin_action_value(
            action_value, self._minimum_action_value, self._maximum_action_value, num_classes)
        action_value_bin = torch.tensor(action_value_bin)
        action_value_bin = F.one_hot(
            action_value_bin, num_classes=num_classes).float()

        item = (item_input, action_value_bin)
        return item


def get_grad_norm(model):
    """
    Calculates and prints the total L2 norm of the gradients of all model parameters.

    Args:
    model (torch.nn.Module): The PyTorch model whose gradients you want to monitor.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            # Square for L2 norm
            param_norm = p.grad.data.norm(2).item()**2
            total_norm += param_norm
        # Take the square root for L2 norm
        total_norm = total_norm**0.5

    return total_norm


def print_model_outputs_for_train_samples(dataset: MyDataset, batch_size: int):
    print("[after training] print_model_outputs_for_train_samples")
    inference_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
    for data in inference_loader:
        (inputs, targets) = data
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        outputs = model(inputs)
        for idx in range(len(inputs[0])):
            print("--------------------")
            print(f"idx: {idx} ")
            input = [inputs[0][idx], inputs[1][idx],
                     inputs[2][idx], inputs[3][idx]]
            target = targets[idx].argmax(dim=-1).item()
            output = outputs[idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(
                list(map(chr, input[0].tolist() + input[1].tolist() + input[2].tolist() + input[3].tolist()))))
            print("target: ")
            print(target)
            print("output: ")
            print(output)
        del inputs
        del targets
        del outputs
        # Only check first batch.
        break


def print_current_state(input_state, current_state):
    input_state_text, full_move_counter, current_state_text = get_state_texts(
        input_state, current_state)

    print(input_state_text)
    print(full_move_counter)
    print(current_state_text)


model = DecoderOnlyTransformerModel(
    vocab_size, d_model, d_ff,  dropout, num_heads, context_size, num_layers, num_classes, device)
# RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
# torch.compile does not work on the NVIDIA P100
# torch.compile works on runpod.io with a NVIDIA A40
# model = torch.compile(model)
model.to(device)

puzzle_train_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "train")

puzzle_test_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "test")


def print_train_examples(train_action_examples):
    print("Train Examples")
    print(len(train_action_examples))
    for (idx, example) in enumerate(train_action_examples):
        print("---------------------------")
        print("Example: " + str(idx))
        sample_input = example[0]
        sample_target = example[1]

        print("sample_input")
        print(tokens_to_text(sample_input))
        print("sample_target")
        print(sample_target)


def train(dataset: MyDataset, batch_size: int, shuffle_train_samples: bool, step: int):
    model.train()
    criterion = nn.CrossEntropyLoss()
    model_total_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trained_train_samples = dataset.__len__()
    num_steps = num_epochs * math.ceil(trained_train_samples / batch_size)

    print(f"parameters: {model_total_params}")
    print(f"trained_train_samples {trained_train_samples}")
    print(f"batch_size {batch_size}")
    print(f"num_epochs {num_epochs}")
    print(f"num_steps {num_steps}")

    steps = []
    losses = []

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_samples, num_workers=8)

    for epoch in range(num_epochs):
        print("Starting epoch")
        for data in train_loader:
            optimizer.zero_grad()
            (inputs, targets) = data
            inputs = [t.to(device) for t in inputs]
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            loss = loss.cpu().item()

            print(
                f"Step: {step}/{num_steps}  epoch: {epoch}  loss: {loss:.8f}")
            steps.append(step)
            losses.append(loss)
            del inputs
            del targets
            del outputs
            step += 1

    return step, steps, losses


def solve_puzzle_example_auto_regressive(input_state, current_state):
    # Note that we can't put the model in evaluation mode because Dropout is important for
    # the model to generalize well during inference according to my tests.
    model.eval()
    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(input_state, current_state)

    # Each cell is allowed to change exactly once.
    for _ in range(puzzle_width * puzzle_height):
        best_next_state = None
        best_action_value = None
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)

        for candidate_action in candidate_actions:
            next_state = copy.deepcopy(current_state)
            row = candidate_action.row()
            col = candidate_action.col()
            cell_value = candidate_action.cell_value()
            next_state[row][col].set_value(cell_value)
            input_tokens = tokenize_sample_input(
                input_state, current_state, candidate_action)
            print("input_text")
            print(tokens_to_text(input_tokens))
            # TODO test all actions in one batch
            inputs = list(map(lambda tensor: tensor.unsqueeze(0),
                          make_sample_tensor(input_tokens)))
            print("inputs size")
            print(inputs[0].size())
            inputs = [t.to(device) for t in inputs]
            outputs = model(inputs)
            print("outputs size")
            print(outputs.size())
            print("outputs")
            print(outputs)
            action_value = outputs[0].argmax(dim=-1).item()
            print(
                f"Testing action  row: {row}  col: {col}  cell_value: {cell_value} action_value: {action_value}")
            if best_action_value == None or action_value > best_action_value:
                best_next_state = next_state
                best_action_value = action_value
            del inputs
            del outputs
        current_state = best_next_state
        print(f"best_next_state with {best_action_value}")
        print("AUTO-REGRESSIVE wannabe AGI megabot current state")
        print_current_state(input_state, current_state)
    return current_state


print("puzzle_train_examples")
print(len(puzzle_train_examples))


def generate_samples(train_dataset_path: str):
    writer = FileStorageWriter(train_dataset_path)
    generated_samples = 0
    while generated_samples < total_train_samples:
        samples = generate_train_action_examples(
            puzzle_train_examples, cell_value_size)
        generated_samples += len(samples)
        writer.append(samples)

    if stop_after_generating_samples:
        sys.exit(42)


if generate_train_samples:
    generate_samples(train_dataset_path)

if load_model:
    print("Loading model")
    state_dict = torch.load(model_file_path, weights_only=True)
    model.load_state_dict(state_dict)

if train_model:
    print("Training model")
    # Create a dataset.
    dataset = MyDataset(train_dataset_path)

    step = 0
    step, steps, losses = train(
        dataset, batch_size, shuffle_train_samples, step)

    if print_model_outputs:
        print_model_outputs_for_train_samples(dataset, batch_size)

    df = pd.DataFrame(data={'step': steps, 'loss': losses})
    df.to_csv('step_loss.csv', index=False)
    torch.save(model.state_dict(),
               model_file_path)


def apply_puzzle_action_value_policy(puzzle_examples):
    for example_input, example_target in puzzle_examples:
        print("example")
        example_input = get_starting_current_state(
            example_input, cell_value_size, input_gen_mode)
        current_state = get_starting_current_state(
            example_target, cell_value_size, current_gen_mode)
        output_state = solve_puzzle_example_auto_regressive(
            example_input, current_state)
        print("final output_state")
        print_current_state(example_input, output_state)
        # TODO make the code work to print the example_target.
        # print("Expected output")
        # print_current_state(
        # example_input, example_target)
        # TODO don't break
        break


# Check if the auto-regressive inference AI is able to predict the output for the train examples.
apply_puzzle_action_value_policy(puzzle_train_examples)


def infer_action_value(model, input_text):
    inputs = make_sample_tensor(input_text).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = model(inputs)
    action_value = outputs[0].argmax(dim=-1).item()
    return action_value


def print_inferred_action_value(model, input_text):
    action_value = infer_action_value(model, input_text)
    print(f"action_value: {action_value}")


# TODO check if the auto-regressive inference AI is able to predict the output for the test example.
# apply_puzzle_action_value_policy(puzzle_test_examples)
