# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-23T13:47:56.066533Z","iopub.execute_input":"2024-07-23T13:47:56.067166Z","iopub.status.idle":"2024-07-23T13:47:56.071486Z","shell.execute_reply.started":"2024-07-23T13:47:56.067134Z","shell.execute_reply":"2024-07-23T13:47:56.070491Z"}}
# Author: Sebastien Boisvert <sebhtml@protonmail.com>
# Git repository: https://github.com/sebhtml/Arc-Prize-2024-sebhtml

# References
# - TODO model should take in input (input_state, current_state)
# - TODO fix action_Value predictor
# - TODO improve stopping criterion in auto-regressive AI
# - TODO implement rotations
# - TODO implement translations

# The model predicts (action_cell, action_value) (write <action_value> to cell <action_cell>)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-23T13:47:56.073115Z","iopub.execute_input":"2024-07-23T13:47:56.073401Z","iopub.status.idle":"2024-07-23T13:47:56.082009Z","shell.execute_reply.started":"2024-07-23T13:47:56.073372Z","shell.execute_reply":"2024-07-23T13:47:56.081089Z"}}
# https://www.kaggle.com/code/sebastien/arc-prize-2024-sebhtml/edit

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import sys
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-23T13:47:56.083820Z","iopub.execute_input":"2024-07-23T13:47:56.084317Z","iopub.status.idle":"2024-07-23T13:47:56.106060Z","shell.execute_reply.started":"2024-07-23T13:47:56.084283Z","shell.execute_reply":"2024-07-23T13:47:56.105176Z"}}
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

selected_puzzle_id = "3aa6fb7a"
context_size = 128
puzzle_width = 7
puzzle_height = 7
vision_width = 7
vision_height = 7
num_classes = 10
d_model = 256
num_heads = 8
dropout = 0.1
num_layers = 16
ascii_printable_size = 128
action_value_bins = 50
batch_size = 256
shuffle = True
lr = 0.001
num_epochs = 6


def make_input_text(current_state, cell, new_value):
    # TODO use / to separate rows
    current_state_text = "".join(map(str, current_state))
    text = "<|current_state|>" + "\n" + current_state_text + "\n" + \
        "<|action|>" + "\n" + str(cell) + " " + str(new_value) + "\n"
    return text


def get_winning_cells(example_output, current_state):
    winning_cells = 0
    for i in range(len(example_output)):
        if example_output[i] == current_state[i]:
            winning_cells += 1
    return winning_cells


def generate_action_examples(puzzle_example):
    (example_input, example_output) = puzzle_example
    action_examples = []
    current_state = example_input
    current_action_value = get_winning_cells(example_output, current_state)
    for cell_addr in range(len(current_state)):
        for cell_value in range(num_classes):
            # TODO An action that put a value that is already in a cell is illegal.
            # TODO An action was already attempted in the past is illegal.
            input_text = make_input_text(current_state, cell_addr, cell_value)
            current_state_tmp = current_state.copy()
            current_state_tmp[cell_addr] = cell_value
            action_value = get_winning_cells(example_output, current_state_tmp)
            example = (input_text, action_value)
            action_examples.append(example)
            if action_value > current_action_value:
                current_state = current_state_tmp

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
        example_input = list(itertools.chain(*example_input))
        example_output = list(itertools.chain(*example_output))

        example = (example_input, example_output)
        puzzle_venue_examples.append(example)
    return puzzle_venue_examples


def generate_train_action_examples(puzzle_examples):
    train_examples = []
    for puzzle_example in puzzle_examples:
        for action_example in generate_action_examples(puzzle_example):
            train_examples.append(action_example)
    return train_examples


def make_example_input_tensor(input_text):
    tokens = [*input_text]
    tokens += [' '] * (context_size - len(tokens))
    tokens = list(map(ord, tokens))
    item_input = torch.tensor(tokens).to(device)
    item_input = F.one_hot(
        item_input, num_classes=ascii_printable_size).float()
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
        action_value = torch.tensor(example[1]).to(device)
        action_value = F.one_hot(
            action_value, num_classes=puzzle_width * puzzle_height + 1).float()

        item = (item_input, action_value)
        return item

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-23T13:47:56.138797Z","iopub.execute_input":"2024-07-23T13:47:56.139043Z","iopub.status.idle":"2024-07-23T13:48:15.616393Z","shell.execute_reply.started":"2024-07-23T13:47:56.139021Z","shell.execute_reply":"2024-07-23T13:48:15.615185Z"}}


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NonCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(NonCausalSelfAttentionTransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src_ln = self.ln1(src)
        attn_output, attn_output_weights = self.multihead_attn(
            src_ln, src_ln, src_ln)
        src_and_sa = self.ln2(src + attn_output)
        src_and_sa_and_ffwd = src_and_sa + self.ffwd(src_and_sa)
        return src_and_sa_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, dropout, num_heads):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.embed = nn.Linear(in_features=ascii_printable_size,
                               out_features=d_model, bias=False)
        self.dropout_1 = nn.Dropout(dropout)
        # TODO honours n_layers
        self.blocks = nn.Sequential(
            NonCausalSelfAttentionTransformerBlock(
                d_model, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                d_model, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                d_model, num_heads, dropout),
            NonCausalSelfAttentionTransformerBlock(
                d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(d_model, num_heads, dropout),
        )
        self.ln = nn.LayerNorm(normalized_shape=d_model)

        self.action_value_lin = nn.Linear(
            in_features=d_model,
            out_features=action_value_bins)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src):
        embed = self.embed(src)
        embed_drop = self.dropout_1(embed)
        transformed = self.blocks(embed_drop)
        transformed_ln = self.ln(transformed)
        action_value = self.action_value_lin(transformed_ln)
        softmax = self.softmax(action_value)
        return softmax


def get_grad_norm(model):
    """
    Calculates and prints the total L2 norm of the gradients of all model parameters.

    Args:
    model (torch.nn.Module): The PyTorch model whose gradients you want to monitor.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()**2  # Square for L2 norm
            total_norm += param_norm
        total_norm = total_norm**0.5  # Take the square root for L2 norm

    return total_norm


def print_predicted_action_values():
    for data in train_loader:
        (inputs, targets) = data
        outputs = model(inputs)
        for idx in range(len(inputs)):
            current_state = inputs[idx].argmax(dim=-1)
            target_action_value = targets[idx].argmax(dim=-1).item()
            # outputs = outputs[:, -1, :]
            print("outputs size")
            print(outputs.size())
            # take last row
            output_action_value = outputs[idx][-1].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(list(map(chr, current_state.tolist()))))
            print("target_action_value: " + str(target_action_value))
            print("output_action_value: " + str(output_action_value))


def print_puzzle_state(puzzle_width, puzzle_height, puzzle_output):
    for row in range(puzzle_height):
        print("|", end="")
        for col in range(puzzle_width):
            cell = row * puzzle_width + col
            value = puzzle_output[cell]
            print(f" {value}", end="")
        print(" |")


model = DecoderOnlyTransformerModel(num_classes, d_model, dropout, num_heads)

criterion = nn.CrossEntropyLoss()
model_total_params = sum(p.numel() for p in model.parameters())
print("Model parameters: " + str(model_total_params))
optimizer = AdamW(model.parameters(), lr=lr)

puzzle_train_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "train")
train_action_examples = generate_train_action_examples(puzzle_train_examples)

puzzle_test_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "test")

print("puzzle_train_examples")
print(len(puzzle_train_examples))
dataset = MyDataset(train_action_examples)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

print("Train Examples")
print(len(train_action_examples))
for (idx, example) in enumerate(train_action_examples):
    print("Example: " + str(idx))
    current_state = example[0]
    action_value = example[1]
    print("input")
    print(current_state)
    print("action_value: " + str(action_value))


def train():
    model.to(device)
    num_steps = num_epochs * len(train_action_examples) // batch_size
    for step in range(num_steps):
        for data in train_loader:
            optimizer.zero_grad()
            (inputs, targets) = data
            outputs = model(inputs)
            # Take last row
            outputs = outputs[:, -1, :]
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            grad_l2_norm = get_grad_norm(model)
            print(
                f"Step: {step + 1}/{num_steps}  grad_norm: {grad_l2_norm:.8f}  loss: {loss:.8f}")


def solve_puzzle_example_auto_regressive(input_state, current_state):
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print_puzzle_state(puzzle_width, puzzle_height, input_state)
    print("current_state on entry")
    print_puzzle_state(puzzle_width, puzzle_height, current_state)

    for iteration in range(10):
        best_cell_addr = None
        best_cell_value = None
        best_action_value = None
        for cell_addr in range(len(current_state)):
            for cell_value in range(num_classes):
                input_text = make_input_text(
                    current_state, cell_addr, cell_value)
                # TODO test all actions in one batch
                inputs = make_example_input_tensor(input_text).unsqueeze(0)
                outputs = model(inputs)
                # action_value = (outputs[:, -1, :][0].argmax(dim=-1).item())
                action_value = (outputs[0].argmax(dim=-1).item())
                # print(f"Testing action  cell_addr: {cell_addr}  cell_value: {cell_value}  action_value: {action_value}")
                if best_action_value == None or action_value > best_action_value:
                    best_action_value = action_value
                    best_cell_addr = cell_addr
                    best_cell_value = cell_value
        current_state = current_state.copy()
        current_state[best_cell_addr] = best_cell_value
        print("current_state after motor action")
        print_puzzle_state(puzzle_width, puzzle_height, current_state)
    return current_state


train()

# TODO predicted action-values currently suck. Fix it. Maybe it's because most of the moves are incorrect (46) so the model outputs 46 and like 95% of the
# predictions are OK.
print("[after training] print_predicted_actions")
print_predicted_action_values()


def do_auto_regressive_prediction():
    for puzzle_train_example_input, puzzle_train_example_output in puzzle_train_examples:
        print("train example")
        output = solve_puzzle_example_auto_regressive(
            puzzle_train_example_input, puzzle_train_example_input)
        print("Expected output")
        print_puzzle_state(puzzle_width, puzzle_height,
                           puzzle_train_example_output)

    for puzzle_test_example_input, puzzle_test_example_output in puzzle_test_examples:
        print("test example")
        output = solve_puzzle_example_auto_regressive(
            puzzle_test_example_input, puzzle_test_example_input)
        print("Expected output")
        print_puzzle_state(puzzle_width, puzzle_height,
                           puzzle_test_example_output)
