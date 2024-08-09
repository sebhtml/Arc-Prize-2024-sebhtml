# Hardware used:

# Legion
# CPU: AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
# GPU: NVIDIA GeForce RTX 4060 8188MiB
# RAM: MemTotal:       32023592 kB

# Kaggle
# CPU:
# GPU: NVIDIA P100
# RAM:

# - TODO make TODO.md file
# - TODO create loss.csv
# - TODO make logger work
# - TODO add class add class Experience with (s, a, r, s')
# - TODO add class QLearningState
# - TODO add class QLearningActionValue
# - TODO use xformers from Meta Platforms.
# - TODO honours n_layers
# - TODO implement translations
# - TODO implement rotations
# - TODO check if the auto-regressive inference AI is able to predict the output for the test example.

# https://www.kaggle.com/code/sebastien/arc-prize-2024-sebhtml/edit

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# This software used reinforcement learning.
# It uses Q-learning.
# See https://en.wikipedia.org/wiki/Q-learning

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import hashlib
import json
import torch
import numpy as np
import copy
import math
import random
# from rotary_embedding_torch import RotaryEmbedding
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
import os
import sys
import itertools
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='/kaggle/working/learning.log',
                    encoding='utf-8', level=logging.DEBUG)
logging.info("Created log file.")

print(f"torch.cuda.is_available {torch.cuda.is_available()}")
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
context_size = 144  # 128 + 16
cell_value_size = 10
puzzle_width = 7
puzzle_height = 7
vision_width = 7
vision_height = 7
hidden_size = 256
ffn_size = 256
num_classes = 50
shuffle = True
num_heads = 8
dropout = 0.1
num_layers = 4
vocab_size = 128
batch_size = 512
lr = 0.0001
weight_decay = 0.01
num_epochs = 400
padding_char = ' '


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


def state_to_text(state) -> str:
    return "\n".join(map(lambda row: "".join(map(str, row)), state))


def make_state_text(input_state, current_state, action: QLearningAction) -> str:
    """
    Q-network
    Q(s, a)
    - s contains the state which is (input_state, current_state)
    - a contains the action which is (next_state)
    """
    input_state_text = state_to_text(input_state)
    current_state_text = state_to_text(current_state)

    # state s
    s = ""
    s += "<|inp|>" + "\n"
    s += input_state_text + "\n"

    s += "<|cur|>" + "\n"
    s += current_state_text + "\n"

    # action a
    a = ""
    a += "<|act|>" + "\n"
    a += str(action.row()).rjust(2, padding_char)
    a += " "
    a += str(action.col()).rjust(2, padding_char)
    a += " "
    a += str(action.cell_value())
    a += "\n"

    text = s + a

    return text


def get_winning_cells(example_output, next_state):
    """
    Q(s, a)
    Count the number of correct cells.
    """
    winning_cells = 0
    for row in range(len(next_state)):
        for col in range(len(next_state[row])):
            if example_output[row][col] == next_state[row][col]:
                winning_cells += 1
    return winning_cells


def get_starting_current_state(example_output, cell_value_size):
    current_state = copy.deepcopy(example_output)
    # Clear state
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            current_state[row][col] = random.randrange(0, cell_value_size)
    return current_state


def generate_cell_actions(current_state, cell_value_size, example_output) -> list[QLearningAction]:
    """
    It is illegal to assign a value to a cell if that cell already has this value.
    """
    candidate_actions = []
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # TODO remove this restriction
            # The action cell value can not change the current cell value if the current cell value is the target cell value.
            if current_state[row][col] == example_output[row][col]:
                continue
            for action_cell_value in range(cell_value_size):
                # TODO remove this restriction
                # The action cell value can not be the current cell value.
                if current_state[row][col] == action_cell_value:
                    continue
                if action_cell_value != example_output[row][col]:
                    continue
                action = QLearningAction(row, col, action_cell_value)
                candidate_actions.append(action)
    np.random.shuffle(candidate_actions)
    return candidate_actions


def generate_action_examples(puzzle_example, cell_value_size):
    (example_input, example_output) = puzzle_example
    action_examples = []
    current_state = get_starting_current_state(example_output, cell_value_size)

    while current_state != example_output:
        best_next_state = None
        best_action_value = None
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size, example_output)

        for candidate_action in candidate_actions:
            next_state = copy.deepcopy(current_state)
            row = candidate_action.row()
            col = candidate_action.col()
            cell_value = candidate_action.cell_value()
            next_state[row][col] = cell_value
            input_text = make_state_text(
                example_input, current_state, candidate_action)
            action_value = get_winning_cells(
                example_output, next_state)
            example = (input_text, action_value)
            action_examples.append(example)
            if best_action_value == None or action_value > best_action_value:
                best_next_state = next_state
                best_action_value = action_value

        current_state = best_next_state

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


def generate_train_action_examples(puzzle_examples, cell_value_size):
    train_examples = []
    for puzzle_example in puzzle_examples:
        for _ in range(32):
            action_examples = generate_action_examples(
                puzzle_example, cell_value_size)
            train_examples += action_examples
    return train_examples


def make_sample_tensor(input_text):
    tokens = [*input_text]
    # Add padding
    tokens += [padding_char] * (context_size - len(tokens))
    tokens = list(map(ord, tokens))
    item_input = torch.tensor(tokens)
    return item_input


def make_sample_text(tensor):
    the_list = tensor.tolist()
    text = "".join(list(map(chr, the_list)))
    return text


class MyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        input_text = example[0]
        item_input = make_sample_tensor(input_text)

        action_value = example[1]
        action_value = torch.tensor(action_value)
        action_value = F.one_hot(
            action_value, num_classes=num_classes).float()

        item = (item_input, action_value)
        return item


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


class PositionalEncoding(nn.Module):
    """
    Generates positional encoding for a given sequence length and embedding dimension.

    Args:
        max_seq_len: Maximum sequence length.
        embed_dim: Embedding dimension.

    Returns:
        Positional encoding tensor.
    """

    def __init__(self, max_seq_len, embed_dim):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(gpu_device)
        self.pe = pe

    def forward(self, x):
        return x + self.pe


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, ffn_size, dropout, num_heads, context_size):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=hidden_size)
        self.pos_encoding = PositionalEncoding(context_size, hidden_size)
        # TODO use RotaryEmbedding
        # self.rotary_emb = RotaryEmbedding(dim=hidden_size)

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
            # NonCausalSelfAttentionTransformerBlock(
            # hidden_size, ffn_size,  num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            # hidden_size, ffn_size, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            # hidden_size, ffn_size, num_heads, dropout),
            # NonCausalSelfAttentionTransformerBlock(
            # hidden_size, ffn_size, num_heads, dropout)
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        # TODO apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)
        # see https://github.com/lucidrains/rotary-embedding-torch
        # x = self.rotary_emb(x)
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
        inputs = inputs.to(gpu_device)
        targets = targets.to(gpu_device)
        outputs = model(inputs)
        print("inputs size")
        print(inputs.size())
        for idx in range(len(inputs)):
            print("--------------------")
            print(f"idx: {idx} ")
            input = inputs[idx]
            target = targets[idx].argmax(dim=-1).item()
            output = outputs[idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(list(map(chr, input.tolist()))))
            print("target: ")
            print(target)
            print("output: ")
            print(output)
        del inputs
        del targets
        del outputs
        # Only check first batch.
        break


def print_puzzle_state(puzzle_width, puzzle_height, puzzle_state):
    for row in range(puzzle_height):
        print("|", end="")
        for col in range(puzzle_width):
            value = puzzle_state[row][col]
            print(f" {value}", end="")
        print(" |")


model = DecoderOnlyTransformerModel(
    vocab_size, hidden_size, ffn_size,  dropout, num_heads, context_size)
# RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
# Does not work on the NVIDIA P100
# model = torch.compile(model)
model.to(gpu_device)

puzzle_train_examples = load_puzzle_examples(
    "training", selected_puzzle_id, "train")
train_action_examples = generate_train_action_examples(
    puzzle_train_examples, cell_value_size)

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
        print(sample_input)
        print("sample_target")
        print(sample_target)


# Create a dataset.
dataset = MyDataset(train_action_examples)

# Create a data loader.
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
            inputs = inputs.to(gpu_device)
            targets = targets.to(gpu_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            grad_l2_norm = get_grad_norm(model)
            loss = loss.cpu().item()
            # logger.info(
            print(
                f"Step: {step}/{num_steps}  epoch: {epoch}  grad_norm: {grad_l2_norm:.8f}  loss: {loss:.8f}")
            steps.append(step)
            losses.append(loss)
            del inputs
            del targets
            del outputs
    df = pd.DataFrame(data={'step': steps, 'loss': losses})
    df.to_csv('step_loss.csv', index=False)


def solve_puzzle_example_auto_regressive(input_state, current_state):
    model.eval()
    print("AUTO-REGRESSIVE wannabe AGI megabot")
    print("input_state")
    print_puzzle_state(puzzle_width, puzzle_height, input_state)
    print("current_state on entry")
    print_puzzle_state(puzzle_width, puzzle_height, current_state)

    # TODO improve stopping criterion in auto-regressive AI
    for _ in range(10):
        best_next_state = None
        best_action_value = None
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)

        for candidate_action in candidate_actions:
            next_state = copy.deepcopy(current_state)
            row = candidate_action.row()
            col = candidate_action.col()
            cell_value = candidate_action.cell_value()
            next_state[row][col] = cell_value
            input_text = make_state_text(
                input_state, current_state, candidate_action)
            print("input_text")
            print(input_text)
            # TODO test all actions in one batch
            inputs = make_sample_tensor(input_text).unsqueeze(0)
            print("inputs size")
            print(inputs.size())
            inputs = inputs.to(gpu_device)
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
        print_puzzle_state(puzzle_width, puzzle_height, current_state)
        # TODO don't break.
        break
    return current_state


print("puzzle_train_examples")
print(len(puzzle_train_examples))

print("train_action_examples")
print(len(train_action_examples))

print("train_action_examples")
print_train_examples(train_action_examples)


train()

print("[after training] print_model_outputs")
print_model_outputs()


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
        # TODO don't break
        break

#
# apply_puzzle_action_value_policy(puzzle_train_examples)

# TODO check if the auto-regressive inference AI is able to predict the output for the test example.
# apply_puzzle_action_value_policy(puzzle_test_examples)
