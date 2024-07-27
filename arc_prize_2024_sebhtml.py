# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:41.982336Z","iopub.execute_input":"2024-07-24T13:09:41.982722Z","iopub.status.idle":"2024-07-24T13:09:41.987008Z","shell.execute_reply.started":"2024-07-24T13:09:41.982694Z","shell.execute_reply":"2024-07-24T13:09:41.986109Z"}}
# Author: Sebastien Boisvert <sebhtml@protonmail.com>
# Git repository: https://github.com/sebhtml/Arc-Prize-2024-sebhtml

# References
# - TODO implement rotations
# - TODO implement translations

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:41.992087Z","iopub.execute_input":"2024-07-24T13:09:41.992721Z","iopub.status.idle":"2024-07-24T13:09:42.000953Z","shell.execute_reply.started":"2024-07-24T13:09:41.992680Z","shell.execute_reply":"2024-07-24T13:09:42.000091Z"}}
# https://www.kaggle.com/code/sebastien/arc-prize-2024-sebhtml/edit

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import torch
import numpy as np
import copy
import math
import random
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import sys
import itertools

print(f"torch.cuda.is_available {torch.cuda.is_available()}")
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:42.002806Z","iopub.execute_input":"2024-07-24T13:09:42.003088Z","iopub.status.idle":"2024-07-24T13:09:42.027236Z","shell.execute_reply.started":"2024-07-24T13:09:42.003065Z","shell.execute_reply":"2024-07-24T13:09:42.026406Z"}}
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

# Model configuration
selected_puzzle_id = "3aa6fb7a"
context_size = 192
cell_value_size = 10
puzzle_width = 7
puzzle_height = 7
vision_width = 7
vision_height = 7
hidden_size = 256
ffn_size = 384
num_heads = 8
dropout = 0.1  # TODO change dropout to 0.5
num_layers = 8
vocab_size = 128
num_classes = 50
batch_size = 512
shuffle = True
lr = 1e-3  # TODO change lr to 1e-4
num_epochs = 100 # TODO reduce number of epochs.
padding_char = '.'


def state_to_text(state):
    return "/".join(map(lambda row: "".join(map(str, row)), state))


def make_input_text(input_state, current_state, row, col, new_value):
    input_state_text = state_to_text(input_state)
    current_state_text = state_to_text(current_state)
    next_state = copy.deepcopy(current_state)
    next_state[row][col] = new_value
    next_state_text = state_to_text(next_state)
    text = ""
    text += "<|input..|>" + "\n" + input_state_text + "\n"
    text += "<|current|>" + "\n" + current_state_text + "\n"
    text += "<|next...|>" + "\n" + next_state_text + "\n"
    return text


def get_winning_cells(example_output, current_state):
    winning_cells = 0
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            if example_output[row][col] == current_state[row][col]:
                winning_cells += 1
    return winning_cells


def get_starting_current_state(example_output):
    current_state = copy.deepcopy(example_output)
    # Clear state
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            current_state[row][col] = 0
    return current_state


def generate_cell_actions(current_state, cell_value_size):
    """
    It is illegal to assign a value to a cell if that cell already has this value.
    """
    candidate_cell_addrs = []
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            for cell_value in range(cell_value_size):
                if current_state[row][col] != cell_value:
                    candidate_cell_addrs.append([row, col, cell_value])
    np.random.shuffle(candidate_cell_addrs)
    return candidate_cell_addrs


def generate_action_examples(puzzle_example):
    (example_input, example_output) = puzzle_example
    action_examples = []
    current_state = get_starting_current_state(example_output)
    current_action_value = None

    # Make a list of incorrect cells.
    candidate_cell_addrs = generate_cell_actions(
        current_state, cell_value_size)

    for row, col, cell_value in candidate_cell_addrs:
        input_text = make_input_text(
            example_input, current_state, row, col, cell_value)
        next_state = copy.deepcopy(current_state)
        next_state[row][col] = cell_value
        action_value = get_winning_cells(example_output, next_state)
        example = (input_text, action_value)
        action_examples.append(example)
        if current_action_value is None or action_value > current_action_value:
            current_state = next_state
            current_action_value = action_value

    return action_examples


def get_puzzle_solution(venue, puzzle_id):
    solutions_file = f"/kaggle/input/arc-prize-2024/arc-agi_{venue}_solutions.json"
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
    challenges_file = f"/kaggle/input/arc-prize-2024/arc-agi_{venue}_challenges.json"
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
    return puzzle_venue_examples


def generate_train_action_examples(puzzle_examples):
    train_examples = []
    for puzzle_example in puzzle_examples:
        for _ in range(32):
            train_examples += generate_action_examples(puzzle_example)
    return train_examples


def make_example_input_tensor(input_text):
    tokens = [*input_text]
    # Add padding
    tokens += [padding_char] * (context_size - len(tokens))
    tokens = list(map(ord, tokens))
    item_input = torch.tensor(tokens)
    return item_input


class MyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example[0]
        item_input = make_example_input_tensor(input_text)
        action_value = torch.tensor(example[1])
        action_value = F.one_hot(
            action_value, num_classes=puzzle_width * puzzle_height + 1).float()

        item = (item_input, action_value)
        return item

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:42.089700Z","iopub.execute_input":"2024-07-24T13:09:42.090047Z","iopub.status.idle":"2024-07-24T13:09:54.926456Z","shell.execute_reply.started":"2024-07-24T13:09:42.090021Z","shell.execute_reply":"2024-07-24T13:09:54.925515Z"}}


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, hidden_size, ffn_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NonCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(self, hidden_size, ffn_size, num_heads, dropout):
        super(NonCausalSelfAttentionTransformerBlock, self).__init__()
        # TODO Re-implement Multihead attention with Rotary Positional Embedding (ROPE)
        # https://www.kaggle.com/code/aeryss/rotary-postional-encoding-rope-pytorch
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout, batch_first=True)
        self.ffwd = FeedForward(hidden_size, ffn_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, src):
        src_ln = self.norm1(src)
        # Self-attention
        attn_output, attn_output_weights = self.attn(
            src_ln, src_ln, src_ln)
        src_and_attn = self.norm2(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffwd(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, ffn_size, dropout, num_heads, num_classes):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=hidden_size)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout_1 = nn.Dropout(dropout)
        # TODO honours n_layers
        self.blocks = nn.Sequential(
            NonCausalSelfAttentionTransformerBlock(
                hidden_size, ffn_size, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                hidden_size, ffn_size, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                hidden_size, ffn_size, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                hidden_size, ffn_size, num_heads, dropout)
            # TODO enable more layers
            # NonCausalSelfAttentionTransformerBlock(
            #    hidden_size, ffn_size,  num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            #    hidden_size, ffn_size, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            #    hidden_size, ffn_size, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            #    hidden_size, ffn_size, num_heads, dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Linear(
            in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = self.embed(x)
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


def print_predicted_action_values():
    for data in train_loader:
        (inputs, targets) = data
        inputs = inputs.to(gpu_device)
        targets = targets.to(gpu_device)
        outputs = model(inputs)
        for idx in range(len(inputs)):
            print(f"idx: {idx} ")
            current_state = inputs[idx]
            target_action_value = targets[idx].argmax(dim=-1).item()
            output_action_value = outputs[idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(list(map(chr, current_state.tolist()))))
            print("target_action_value: " + str(target_action_value))
            print("output_action_value: " + str(output_action_value))
        del inputs
        del targets
        del outputs


def print_puzzle_state(puzzle_width, puzzle_height, puzzle_state):
    for row in range(puzzle_height):
        print("|", end="")
        for col in range(puzzle_width):
            value = puzzle_state[row][col]
            print(f" {value}", end="")
        print(" |")


model = DecoderOnlyTransformerModel(
    vocab_size, hidden_size, ffn_size,  dropout, num_heads, num_classes)
# RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
# Does not work on the NVIDIA P100
# model = torch.compile(model)
model.to(gpu_device)

puzzle_train_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "train")
train_action_examples = generate_train_action_examples(puzzle_train_examples)

puzzle_test_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "test")


def print_train_examples():
    print("Train Examples")
    print(len(train_action_examples))
    for (idx, example) in enumerate(train_action_examples):
        print("Example: " + str(idx))
        current_state = example[0]
        action_value = example[1]
        print("input")
        print(current_state)
        print("action_value: " + str(action_value))


dataset = MyDataset(train_action_examples)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train():
    model.train()
    criterion = nn.CrossEntropyLoss()
    model_total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: " + str(model_total_params))
    # TODO increase weight_decay to do L2 regularization.
    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"num_epochs {num_epochs}")
    num_steps = num_epochs * math.ceil(len(train_action_examples) / batch_size)
    step = 0
    for _ in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            (inputs, targets) = data
            inputs = inputs.to(gpu_device)
            targets = targets.to(gpu_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            grad_l2_norm = get_grad_norm(model)
            print(
                f"Step: {step}/{num_steps}  grad_norm: {grad_l2_norm:.8f}  loss: {loss:.8f}")
            del inputs
            del targets
            del outputs


def solve_puzzle_example_auto_regressive(input_state, current_state):
    model.eval()
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print_puzzle_state(puzzle_width, puzzle_height, input_state)
    print("current_state on entry")
    print_puzzle_state(puzzle_width, puzzle_height, current_state)

    # TODO improve stopping criterion in auto-regressive AI
    for _ in range(10):
        best_cell_row = None
        best_cell_col = None
        best_cell_value = None
        best_action_value = None
        candidate_cell_addrs = generate_cell_actions(
            current_state, cell_value_size)

        for row, col, cell_value in candidate_cell_addrs:
            input_text = make_input_text(
                input_state, current_state, row, col, cell_value)
            # TODO test all actions in one batch
            inputs = make_example_input_tensor(input_text).unsqueeze(0)
            inputs = inputs.to(gpu_device)
            outputs = model(inputs)
            action_value = outputs[0].argmax(dim=-1).item()
            # print(
            # f"Testing action  cell_addr: {cell_addr}  cell_value: {cell_value}  action_value: {action_value}")
            if best_action_value == None or action_value > best_action_value:
                best_action_value = action_value
                best_cell_row = row
                best_cell_col = col
                best_cell_value = cell_value
            del inputs
            del outputs
        print(
            f"Applying action  best_cell_row: {best_cell_row}  best_cell_col: {best_cell_col}  cell_value: {best_cell_value}  action_value: {best_action_value}")
        current_state = copy.deepcopy(current_state)
        current_state[best_cell_row][best_cell_col] = best_cell_value
        print("current_state after motor action")
        print_puzzle_state(puzzle_width, puzzle_height, current_state)
    return current_state


print("puzzle_train_examples")
print(len(puzzle_train_examples))

print("train_action_examples")
print(len(train_action_examples))

train()

print("[after training] print_predicted_actions")
#print_predicted_action_values()


def apply_puzzle_action_value_policy(puzzle_examples):
    for example_input, example_target in puzzle_examples:
        print("example")
        current_state = get_starting_current_state(example_target)
        output_state = solve_puzzle_example_auto_regressive(
            example_input, current_state)
        print("final output_state")
        print_puzzle_state(puzzle_width, puzzle_height, output_state)
        print("Expected output")
        print_puzzle_state(puzzle_width, puzzle_height,
                           example_target)


# TODO make it work on the train examples
apply_puzzle_action_value_policy(puzzle_train_examples)

# TODO check if the auto-regressive inference AI is able to predict the output for the test example.
# apply_puzzle_action_value_policy(puzzle_test_examples)
