# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:41.982336Z","iopub.execute_input":"2024-07-24T13:09:41.982722Z","iopub.status.idle":"2024-07-24T13:09:41.987008Z","shell.execute_reply.started":"2024-07-24T13:09:41.982694Z","shell.execute_reply":"2024-07-24T13:09:41.986109Z"}}
# Author: Sebastien Boisvert <sebhtml@protonmail.com>
# Git repository: https://github.com/sebhtml/Arc-Prize-2024-sebhtml

# References
# - TODO fix bug that causes the model to always generate action_value 44.
# - TODO generate more examples
# - TODO improve stopping criterion in auto-regressive AI
# - TODO implement rotations
# - TODO implement translations

# The model predicts (action_cell, action_value) (write <action_value> to cell <action_cell>)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

selected_puzzle_id = "3aa6fb7a"
context_size = 160
colors = 10
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
# TODO rename action_value_bins to num_classes
action_value_bins = 50
batch_size = 64
shuffle = True
lr = 1e-3
num_epochs = 30
PADDING_CHAR = '.'


def make_input_text(initial_state, current_state, cell, new_value):
    # TODO use / to separate rows
    initial_state_text = "".join(map(str, initial_state))
    current_state_text = "".join(map(str, current_state))
    text = ""
    text += "<|initial_state|>" + "\n" + initial_state_text + "\n"
    text += "<|current_state|>" + "\n" + current_state_text + "\n"
    text += "<|action|>" + "\n" + \
        str(cell).ljust(2, PADDING_CHAR) + " " + str(new_value) + "\n"
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
    current_state = example_output.copy()
    # Clear initial current state
    for cell_addr in range(len(current_state)):
        current_state[cell_addr] = random.randint(0, colors - 1)
    current_action_value = get_winning_cells(example_output, current_state)

    # make a list of incorrect cells.
    candidate_cell_addrs = []
    for cell_addr in range(len(current_state)):
        if current_state[cell_addr] != example_output[cell_addr]:
            candidate_cell_addrs.append(cell_addr)

    np.random.shuffle(candidate_cell_addrs)

    for cell_addr in candidate_cell_addrs:
        candidate_cell_values = list(range(colors))
        np.random.shuffle(candidate_cell_values)
        for cell_value in candidate_cell_values:
            input_text = make_input_text(
                example_input, current_state, cell_addr, cell_value)
            current_state_tmp = current_state.copy()
            current_state_tmp[cell_addr] = cell_value
            action_value = get_winning_cells(example_output, current_state_tmp)
            example = (input_text, action_value)
            action_examples.append(example)
            if action_value > current_action_value:
                current_state = current_state_tmp
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
        # TODO keep the rows instead of using chaining
        example_input = list(itertools.chain(*example_input))
        example_output = list(itertools.chain(*example_output))

        example = (example_input, example_output)
        puzzle_venue_examples.append(example)
    return puzzle_venue_examples


def generate_train_action_examples(puzzle_examples):
    train_examples = []
    for puzzle_example in puzzle_examples:
        for _ in range(1):
            train_examples += generate_action_examples(puzzle_example)
        # TODO don't break
        break
    return train_examples


def make_example_input_tensor(input_text):
    tokens = [*input_text]
    # add padding
    tokens += [PADDING_CHAR] * (context_size - len(tokens))
    tokens = list(map(ord, tokens))
    # TODO call .to(device) in training loop
    item_input = torch.tensor(tokens).to(device)
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
        # TODO call .to(device in training loop)
        action_value = torch.tensor(example[1]).to(device)
        action_value = F.one_hot(
            action_value, num_classes=puzzle_width * puzzle_height + 1).float()

        item = (item_input, action_value)
        return item

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-07-24T13:09:42.089700Z","iopub.execute_input":"2024-07-24T13:09:42.090047Z","iopub.status.idle":"2024-07-24T13:09:54.926456Z","shell.execute_reply.started":"2024-07-24T13:09:42.090021Z","shell.execute_reply":"2024-07-24T13:09:54.925515Z"}}


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
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout, batch_first=True)
        self.ffwd = FeedForward(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src_ln = self.norm1(src)
        # Self-attention
        attn_output, attn_output_weights = self.attn(
            src_ln, src_ln, src_ln)
        src_and_attn = self.norm2(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffwd(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, num_classes, d_model, dropout, num_heads):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=ascii_printable_size,
                                  embedding_dim=d_model)
        self.gap = nn.AdaptiveAvgPool1d(1)

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
        )
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(
            in_features=d_model, out_features=action_value_bins)

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
    model.to(device)
    num_steps = math.ceil(num_epochs * len(train_action_examples) / batch_size)
    step = 0
    print(f"num_steps {num_steps}")
    while step < num_steps:
        for data in train_loader:
            optimizer.zero_grad()
            (inputs, targets) = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            grad_l2_norm = get_grad_norm(model)
            print(
                f"Step: {step}/{num_steps}  grad_norm: {grad_l2_norm:.8f}  loss: {loss:.8f}")


def solve_puzzle_example_auto_regressive(input_state, current_state):
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print("current_state on entry")

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
                action_value = outputs[0].argmax(dim=-1).item()
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


print("puzzle_train_examples")
print(len(puzzle_train_examples))

print("train_action_examples")
print(len(train_action_examples))

train()

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
