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

# - TODO investigate model inference predicted action value using the function print_inferred_action_value.

# - TODO check if the auto-regressive inference AI is able to predict the output for the train examples

# - TODO generate action_examples once and write them to disk

# - TODO add class QLearningState
# - TODO add class QLearningActionValue

# - TODO implement translations
# - TODO implement rotations

# - TODO use "Feed forward mechanisms" from xformers
# - TODO use "Residual paths" from xformers
# - TODO add class add class Experience with (s, a, r, s')

# - TODO check if the auto-regressive inference AI is able to predict the output for the test example.

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
import hashlib
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from xformers.components import MultiHeadDispatch, build_attention
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
import sys
import itertools

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

# Model configuration
selected_puzzle_id = "3aa6fb7a"
context_size = 186  # multiple of 4
cell_value_size = 10
puzzle_width = 7
puzzle_height = 7
d_model = 256
d_ff = 768
num_classes = 128
shuffle = True
num_heads = 8
dropout = 0.1
num_layers = 4
vocab_size = 128  # puzzle_width * puzzle_height * cell_value_size
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
weight_decay = 0.01
discount = 0.99
num_epochs = 5
# Use 1 for development, for production, use 87.
sample_augmentation_multiplier = 87
padding_char = ' '
stop_after_generating_samples = False
load_model = False
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


class SampleInputTokens:
    def __init__(self, input_state, current_state, action):
        self._input_state = input_state
        self._current_state = current_state
        self._action = action


def tokenize_sample_input(input_state, current_state, action: QLearningAction) -> SampleInputTokens:
    """
    Tokenize a sample input for the Q-network Q(s, a).
    Note that:
    - s contains the state which is (input_state, current_state)
    - a contains the action which is (next_state)
    """

    # state s
    input_state_text = ""
    input_state_text += "ini" + "\n"
    input_state_text += input_state_to_text(input_state)

    current_state_text = ""
    current_state_text += "cur" + "\n"
    current_state_text += current_state_to_text(current_state)

    action_text = ""
    action_text += "act" + "\n"
    action_text += action_to_text(current_state, action)

    return SampleInputTokens(
        text_to_tokens(input_state_text),
        text_to_tokens(current_state_text),
        text_to_tokens(action_text)
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def tokens_to_text(sample_input_tokens: SampleInputTokens) -> str:
    tokens: List[int] = sample_input_tokens._input_state + \
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
    # TODO refactor loop to simply count the number of unchanged cells.
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

    # print("DONE")

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


def generate_train_action_examples(puzzle_examples, cell_value_size):
    train_examples = []
    for puzzle_example in puzzle_examples:
        for _ in range(sample_augmentation_multiplier):
            action_examples = generate_action_examples(
                puzzle_example, cell_value_size)
            train_examples += action_examples
    return train_examples


def make_sample_tensor(sample_input_tokens: SampleInputTokens):
    input_tokens: List[int] = sample_input_tokens._input_state + \
        sample_input_tokens._current_state + sample_input_tokens._action
    if len(input_tokens) > context_size:
        raise Exception(
            f"text ({len(input_tokens)} tokens) is too large to fit in context ! Increase context_size ({context_size})")
    item_input = [torch.tensor(sample_input_tokens._input_state),
                  torch.tensor(sample_input_tokens._current_state),
                  torch.tensor(sample_input_tokens._action)
                  ]
    return item_input


def make_sample_text(tensor):
    the_list = tensor.tolist()
    text = "".join(list(map(chr, the_list)))
    return text


class MyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

        # Compute minimum and maximum action value
        action_values = list(map(lambda example: example[1], examples))
        self._minimum_action_value = min(action_values)
        self._maximum_action_value = max(action_values)
        print(f"_minimum_action_value {self._minimum_action_value}")
        print(f"_maximum_action_value {self._maximum_action_value}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        input_tokens = example[0]
        item_input = make_sample_tensor(input_tokens)

        action_value = example[1]
        # convert action_value to { 0, 1, ..., num_classes - 1 }
        # Example:
        # action_value: 3.0
        # _minimum_action_value: -4.0
        # _maximum_action_value: 7.0
        # action_value_bin = (3.0 - -4.0) / (7.0 - -4.0)
        # TODO move this code to a function.
        action_value_bin = math.floor(
            ((action_value - self._minimum_action_value) / (self._maximum_action_value - self._minimum_action_value)) * (num_classes - 1))
        action_value_bin = torch.tensor(action_value_bin)
        action_value_bin = F.one_hot(
            action_value_bin, num_classes=num_classes).float()

        item = (item_input, action_value_bin)
        return item


class SwiGLU(nn.Module):
    """
    swiGLU = (x * U) * swish(x * V)
    Note that in PyTorch, the name of Swish is SILU
    """

    def __init__(self, dim):
        super().__init__()
        self.swish = nn.SiLU()
        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        input = self.input_proj(x)
        gate = self.swish(self.gate_proj(x))
        return input * gate


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.swiglu = SwiGLU(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NonCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(self, hidden_size, ffn_size, num_heads, dropout, context_size):
        super(NonCausalSelfAttentionTransformerBlock, self).__init__()
        my_config = {
            # you can easily make this dependent on a file, sweep,..
            "name": "scaled_dot_product",
            "dropout": dropout,
            "seq_len": context_size,
            "causal": False,
        }
        attention = build_attention(my_config)
        self.attn = MultiHeadDispatch(
            seq_len=context_size,
            dim_model=hidden_size,
            residual_dropout=dropout,
            num_heads=num_heads,
            attention=attention,
            use_rotary_embeddings=True,
        ).to(device)

        self.ffn = FeedForward(hidden_size, ffn_size, dropout)
        self.attention_norm = nn.RMSNorm(hidden_size)
        self.ffn_norm = nn.RMSNorm(hidden_size)

    def forward(self, src):
        src_ln = self.attention_norm(src)
        # Self-attention
        attn_output = self.attn(
            query=src_ln, key=src_ln, value=src_ln)
        src_and_attn = self.ffn_norm(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffn(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, ffn_size, dropout, num_heads, context_size, num_layers):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.input_embed = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=hidden_size)
        self.current_embed = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size)
        self.action_embed = nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=hidden_size)

        self.dropout_1 = nn.Dropout(dropout)
        modules = [NonCausalSelfAttentionTransformerBlock(
            hidden_size, ffn_size, num_heads, dropout, context_size) for _ in range(num_layers)]

        self.blocks = nn.Sequential(*modules)
        self.norm = nn.RMSNorm(hidden_size)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x0 = self.input_embed(x[0])
        x1 = self.current_embed(x[1])
        x2 = self.action_embed(x[2])
        x = torch.cat([x0, x1, x2], dim=1)
        x = x / math.sqrt(d_model)
        embed_drop = self.dropout_1(x)
        transformed = self.blocks(embed_drop)
        transformed_ln = self.norm(transformed)
        last_hidden_state = transformed_ln
        output = self.gap(last_hidden_state.transpose(1, 2))
        output = output.squeeze(2)
        logits = self.classifier(output)
        return logits


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


def print_model_outputs():
    for data in train_loader:
        (inputs, targets) = data
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        outputs = model(inputs)
        for idx in range(len(inputs[0])):
            print("--------------------")
            print(f"idx: {idx} ")
            input = [inputs[0][idx], inputs[1][idx], inputs[2][idx]]
            target = targets[idx].argmax(dim=-1).item()
            output = outputs[idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(
                list(map(chr, input[0].tolist() + input[1].tolist() + input[2].tolist()))))
            print("target: ")
            print(target)
            print("output: ")
            print(output)
        del inputs
        del targets
        del outputs
        # Only check first batch.
        break


def print_current_state(current_state):
    s: str = ""
    current_state_value_text = current_state_to_text(current_state)
    s += "<|cur|>" + "\n"
    s += current_state_value_text + "\n"
    print(s)


model = DecoderOnlyTransformerModel(
    vocab_size, d_model, d_ff,  dropout, num_heads, context_size, num_layers)
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
        # TODO use decode of the Tokenizer.
        print("sample_input")
        print(tokens_to_text(sample_input))
        print("sample_target")
        print(sample_target)


def train():

    model.train()
    criterion = nn.CrossEntropyLoss()
    model_total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: " + str(model_total_params))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"num_epochs {num_epochs}")
    num_steps = num_epochs * math.ceil(len(train_action_examples) / batch_size)
    print(f"num_steps {num_steps}")
    step = 0
    steps = []
    losses = []
    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            (inputs, targets) = data
            inputs = [t.to(device) for t in inputs]
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            loss = loss.cpu().item()
            # current_memory_allocated = torch.cuda.memory_allocated()
            print(
                f"Step: {step}/{num_steps}  epoch: {epoch}  loss: {loss:.8f}")
            steps.append(step)
            losses.append(loss)
            del inputs
            del targets
            del outputs
    df = pd.DataFrame(data={'step': steps, 'loss': losses})
    df.to_csv('step_loss.csv', index=False)


def solve_puzzle_example_auto_regressive(input_state, current_state):
    # Note that we can't put the model in evaluation mode because Dropout is important for
    # the model to generalize well during inference according to my tests.
    model.eval()
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print_current_state(input_state)
    print("current_state on entry")
    print_current_state(current_state)

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
            print(input_tokens)
            # TODO test all actions in one batch
            inputs = make_sample_tensor(input_tokens).unsqueeze(0)
            print("inputs size")
            print(inputs.size())
            inputs = inputs.to(device)
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
        print("current_state after motor action")
        print_current_state(current_state)
    return current_state


print("puzzle_train_examples")
print(len(puzzle_train_examples))


if load_model:
    print("Loading model")
    state_dict = torch.load(model_file_path, weights_only=True)
    model.load_state_dict(state_dict)
else:
    train_action_examples = generate_train_action_examples(
        puzzle_train_examples, cell_value_size)

    print("train_action_examples")
    print(len(train_action_examples))

    print("train_action_examples")
    print_train_examples(train_action_examples)

    if stop_after_generating_samples:
        sys.exit(42)

    # Create a dataset.
    dataset = MyDataset(train_action_examples)

    # Create a data loader.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print("Training model")
    train()
    torch.save(model.state_dict(),
               model_file_path)

    print("[after training] print_model_outputs")
    print_model_outputs()


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
        print_current_state(output_state)
        print("Expected output")
        print_current_state(
            example_target)
        # TODO don't break
        break


# Check if the auto-regressive inference AI is able to predict the output for the train examples.
# apply_puzzle_action_value_policy(puzzle_train_examples)


def infer_action_value(model, input_text):
    inputs = make_sample_tensor(input_text).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = model(inputs)
    action_value = outputs[0].argmax(dim=-1).item()
    return action_value


def print_inferred_action_value(model, input_text):
    action_value = infer_action_value(model, input_text)
    print(f"action_value: {action_value}")


input_text_with_action_value_50 = """<|inp|>
0000000
0800000
0880000
0000880
0000080
0000000
0000000

<|cur|>
0000000
0010000
0880000
0000000
0000180
0000000
0000000

<|cha|>
XXX_XX_
X_X____
XXX____
XXXX___
_XXXXXX
X_XX_XX
XXX_XXX

<|act|>
0  6 7"""

# print_inferred_action_value(model, input_text_with_action_value_50)

input_text_with_action_value_50_modified = """<|inp|>
0000000
0800000
0880000
0000880
0000080
0000000
0000000

<|cur|>
0000000
0010000
0880000
0000000
0000180
0000000
0000000

<|cha|>
XXX_XX_
X_X____
XXX____
XXXX___
_XXXXXX
X_XX_XX
XXX_XXX

<|act|>
0  6 9"""

# print_inferred_action_value(model, input_text_with_action_value_50_modified)

input_text_end_game = """<|inp|>
0000000
0800000
0880000
0000880
0000080
0000000
0000000

<|cur|>
8189120
0016858
7064051
1893332
1007289
0289648
5762399

<|cha|>
XXXXXXX
XXXXXXX
X_XXXXX
XXXXXXX
XXXXXXX
XXXXXXX
XXXXXXX

<|act|>
 2  1 4"""

ex3 = """<|inp|>
0000000
0800000
0880000
0000880
0000080
0000000
0000000

<|cur|>
0000000
0000000
0000000
0000000
0000000
0000000
0000000

<|cha|>
_______
_______
_______
_______
_______
_______
_______

<|act|>
 1  1 8"""

# print_inferred_action_value(model, ex3)


# TODO check if the auto-regressive inference AI is able to predict the output for the test example.
# apply_puzzle_action_value_policy(puzzle_test_examples)
