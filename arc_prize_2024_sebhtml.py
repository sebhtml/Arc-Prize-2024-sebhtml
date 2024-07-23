# %% [code]
# Author: Sebastien Boisvert <sebhtml@protonmail.com>
# Git repository: https://github.com/sebhtml/Arc-Prize-2024-sebhtml

# References
# - TODO output a single output instead of two for cell_row, cell_col, cell_value
# - TODO model should take in input (input_state, current_state)
# - TODO improve stopping criterion in auto-regressive AI
# - TODO use CUDA (NVIDIA P100) on Kaggle
# - TODO implement rotations
# - TODO implement translations

# The model predicts (action_cell, action_value) (write <action_value> to cell <action_cell>)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.299013Z","iopub.execute_input":"2024-07-18T17:58:05.299454Z","iopub.status.idle":"2024-07-18T17:58:05.306369Z","shell.execute_reply.started":"2024-07-18T17:58:05.299420Z","shell.execute_reply":"2024-07-18T17:58:05.305066Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.308582Z","iopub.execute_input":"2024-07-18T17:58:05.309436Z","iopub.status.idle":"2024-07-18T17:58:05.324028Z","shell.execute_reply.started":"2024-07-18T17:58:05.309403Z","shell.execute_reply":"2024-07-18T17:58:05.322739Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.325644Z","iopub.execute_input":"2024-07-18T17:58:05.326104Z","iopub.status.idle":"2024-07-18T17:58:05.336750Z","shell.execute_reply.started":"2024-07-18T17:58:05.326060Z","shell.execute_reply":"2024-07-18T17:58:05.335390Z"}}
selected_puzzle_id = "3aa6fb7a"
puzzle_width = 7
puzzle_height = 7
vision_width = 7
vision_height = 7
num_classes = 10
d_model = 768
num_heads = 12
dropout = 0.1
num_layers = 6
batch_size = 512
shuffle = True
lr = 0.001
num_epochs = 100
num_layers = 12

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.339070Z","iopub.execute_input":"2024-07-18T17:58:05.339468Z","iopub.status.idle":"2024-07-18T17:58:05.349542Z","shell.execute_reply.started":"2024-07-18T17:58:05.339437Z","shell.execute_reply":"2024-07-18T17:58:05.348400Z"}}


def generate_action_examples(puzzle_example):
    (example_input, example_output) = puzzle_example
    action_examples = []
    current_state = example_input
    for i in range(len(example_input)):
        input_pixel = example_input[i]
        output_pixel = example_output[i]
        if input_pixel != output_pixel:
            action_cell = i
            action_value = output_pixel
            example = (current_state, (action_cell, action_value))
            action_examples.append(example)
            # Update current_state
            current_state = current_state.copy()
            current_state[action_cell] = action_value
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


def make_example_input_tensor(puzzle_example_current_state):
    item_input = torch.tensor(puzzle_example_current_state)
    item_input = F.one_hot(item_input, num_classes=num_classes).float()
    return item_input

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.396980Z","iopub.execute_input":"2024-07-18T17:58:05.397548Z","iopub.status.idle":"2024-07-18T17:58:05.406723Z","shell.execute_reply.started":"2024-07-18T17:58:05.397499Z","shell.execute_reply":"2024-07-18T17:58:05.405477Z"}}


class MyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        item_input = make_example_input_tensor(example[0])
        item_output = example[1]

        action_cell = item_output[0]
        action_cell = torch.tensor(action_cell)
        action_cell = F.one_hot(
            action_cell, num_classes=puzzle_width * puzzle_height).float()

        action_value = item_output[1]
        action_value = torch.tensor(action_value)
        action_value = F.one_hot(action_value, num_classes=num_classes).float()

        item_output = (action_cell, action_value)
        item = (item_input, item_output)
        return item

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.409064Z","iopub.execute_input":"2024-07-18T17:58:05.409463Z","iopub.status.idle":"2024-07-18T17:58:05.436922Z","shell.execute_reply.started":"2024-07-18T17:58:05.409420Z","shell.execute_reply":"2024-07-18T17:58:05.435808Z"}}


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
        # print("attn_output")
        # print(attn_output)
        src_and_sa = self.ln2(src + attn_output)
        src_and_sa_and_ffwd = src_and_sa + self.ffwd(src_and_sa)
        return src_and_sa_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, dropout, num_heads):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.embed = nn.Linear(in_features=num_classes,
                               out_features=d_model, bias=False)
        self.dropout_1 = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
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

        self.action_cell_lin = nn.Linear(
            in_features=d_model,
            out_features=puzzle_width * puzzle_height)
        self.action_value_lin = nn.Linear(
            in_features=d_model,
            out_features=num_classes)
        self.action_cell_soft = nn.Softmax(dim=-1)
        self.action_value_soft = nn.Softmax(dim=-1)

    def forward(self, src):
        embed = self.embed(src)
        embed_drop = self.dropout_1(embed)
        transformed = self.blocks(embed_drop)
        transformed_ln = self.ln(transformed)
        action_cell = self.action_cell_lin(transformed_ln)
        action_value = self.action_value_lin(transformed_ln)
        return (action_cell, action_value)
        action_cell = self.action_cell_soft(self.action_cell_lin(reshaped))
        action_value = self.action_value_soft(self.action_value_lin(reshaped))
        return (action_cell, action_value)


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


def print_predicted_actions():
    for data in train_loader:
        (inputs, targets) = data
        print(inputs.size())
        outputs = model(inputs)
        for idx in range(len(inputs)):
            current_state = inputs[idx].argmax(dim=-1)
            target_action_cell = targets[0][idx].argmax(dim=-1).item()
            target_action_value = targets[1][idx].argmax(dim=-1).item()
            output_action_cell = outputs[0][:, -
                                            1, :][idx].argmax(dim=-1).item()
            output_action_value = outputs[1][:, -
                                             1, :][idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("current_state")
            print_puzzle_state(puzzle_width, puzzle_height, current_state)
            print("target_action_cell: " + str(target_action_cell))
            print("target_action_value: " + str(target_action_value))
            print("output_action_cell: " + str(output_action_cell))
            print("output_action_value: " + str(output_action_value))


def print_puzzle_state(puzzle_width, puzzle_height, puzzle_output):
    for row in range(puzzle_height):
        print("|", end="")
        for col in range(puzzle_width):
            cell = row * puzzle_width + col
            value = puzzle_output[cell]
            print(f" {value}", end="")
        print(" |")


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-18T17:58:05.438134Z","iopub.execute_input":"2024-07-18T17:58:05.438488Z","iopub.status.idle":"2024-07-18T17:58:05.684885Z","shell.execute_reply.started":"2024-07-18T17:58:05.438459Z","shell.execute_reply":"2024-07-18T17:58:05.683364Z"}}
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
    action_cell = example[1][0]
    action_value = example[1][1]
    print_puzzle_state(puzzle_width, puzzle_height, current_state)
    print("output_action_cell: " + str(action_cell))
    print("output_action_value: " + str(action_value))

global_step = 0
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        (inputs, targets) = data
        outputs = model(inputs)
        # Use only the logits in the last row.
        cell_loss = criterion(outputs[0][:, -1, :], targets[0])
        pixel_loss = criterion(outputs[1][:, -1, :], targets[1])
        loss = cell_loss + pixel_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        grad_l2_norm = get_grad_norm(model)
        print(
            f"Epoch: {epoch + 1} / {num_epochs}  global_step: {global_step + 1}  grad_l2_norm: {grad_l2_norm:.8f}  loss: {loss:.8f}")
        global_step += 1

print("[after training] print_predicted_actions")
print_predicted_actions()


def solve_puzzle_example_auto_regressive(input_state, current_state):
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print_puzzle_state(puzzle_width, puzzle_height, input_state)
    print("current_state on entry")
    print_puzzle_state(puzzle_width, puzzle_height, current_state)

    for i in range(10):
        inputs = make_example_input_tensor(current_state).unsqueeze(0)
        outputs = model(inputs)
        action_cell = outputs[0][:, -1, :][0].argmax(dim=-1).item()
        action_value = (outputs[1][:, -1, :][0].argmax(dim=-1).item())
        current_state = current_state.copy()
        current_state[action_cell] = action_value
        print("current_state after motor action")
        print_puzzle_state(puzzle_width, puzzle_height, current_state)
    return current_state


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
    print_puzzle_state(puzzle_width, puzzle_height, puzzle_test_example_output)
