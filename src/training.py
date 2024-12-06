import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import random
import math
from datetime import datetime, timezone
from typing import List, Tuple
from agent import make_example_tensor, generate_examples
from context import tokens_to_text
from context import ExampleInputTokens
from report import plot_train_loss_graph
from model import DecoderOnlyTransformerModel
from emulator import Emulator


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
    def __init__(self,
                 train_examples: List[Tuple[ExampleInputTokens, float]],
                 context_size: int, num_classes: int):
        self.context_size = context_size
        self.num_classes = num_classes
        self.train_examples = train_examples
        min_value, max_value = -50.0, +50.0
        self._minimum_action_value = min_value
        self._maximum_action_value = max_value

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        example = self.train_examples[idx]

        input_tokens = example[0]
        item_input = make_example_tensor(input_tokens, self.context_size)

        action_value = example[1]
        action_value_bin = bin_action_value(
            action_value, self._minimum_action_value, self._maximum_action_value, self.num_classes)
        action_value_bin = torch.tensor(action_value_bin)
        action_value_bin = F.one_hot(
            action_value_bin, num_classes=self.num_classes).float()

        item = (item_input, action_value_bin)
        return item

    def get_action_value_min_max(self, train_examples: List[Tuple[ExampleInputTokens, float]]) -> Tuple[float, float]:
        min_action_value = min(map(lambda example: example[1], train_examples))
        max_action_value = max(map(lambda example: example[1], train_examples))
        return min_action_value, max_action_value


def trim_list(lst, k):
    """
    keep at most k elements from list lst
    """
    return lst[-k:] if len(lst) > k else lst


def train(
        criterion: CrossEntropyLoss,
        optimizer: AdamW,
        dataset: MyDataset, batch_size: int, shuffle_train_examples: bool, model: DecoderOnlyTransformerModel,
        max_grad_norm: float, device: torch.device,
):
    model.train()

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_examples, num_workers=2, drop_last=True)

    data = next(iter(train_loader))

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

    for t in inputs:
        del t
    del targets
    del outputs

    return loss


def print_train_examples(train_action_examples):
    print("Train Examples")
    print(len(train_action_examples))
    for (idx, example) in enumerate(train_action_examples):
        print("---------------------------")
        print("Example: " + str(idx))
        example_input = example[0]
        example_target = example[1]

        print("example_input")
        print(tokens_to_text(example_input))
        print("example_target")
        print(example_target)


def print_model_outputs_for_train_examples(dataset: MyDataset, batch_size: int, model, device,):
    print("[after training] print_model_outputs_for_train_examples")
    inference_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    for data in inference_loader:
        (inputs, targets) = data
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        outputs = model(inputs)
        for idx in range(len(inputs[0])):
            print("--------------------")
            print(f"idx: {idx} ")
            input = [inputs[0][idx], inputs[1][idx],
                     inputs[2][idx]]
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
        for t in inputs:
            del t
        del targets
        del outputs
        # Only check first batch for now.
        break


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


def train_model_using_experience_replay(
        context_size: int, batch_size: int, device: torch.device,
    model: DecoderOnlyTransformerModel, total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool, lr: float, weight_decay: float,
    max_grad_norm: float, print_model_outputs: bool, save_step_losses: bool,

):
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    emulator = Emulator(cell_value_size)
    max_taken_actions_per_step = 1
    steps = []
    losses = []
    num_steps = 300  # 32000

    experience_replay_data_set = []
    for step in range(num_steps):
        experience_replay_data_set, loss = train_model_with_experience_replay_data_set(
            emulator,
            max_taken_actions_per_step,
            criterion,
            optimizer,
            experience_replay_data_set,
            context_size, batch_size, device, model, total_train_examples,
            puzzle_train_examples, cell_value_size,
            discount, padding_char, num_classes, shuffle_train_examples,
            max_grad_norm, print_model_outputs,
        )

        if loss != None:
            steps.append(step)
            losses.append(loss)
            print(f"Step: {step}/{num_steps}  loss: {loss:.8f}")

    dynamic_time_marker = datetime.now(timezone.utc).isoformat()
    train_loss_csv_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.csv"
    train_loss_png_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.png"

    if save_step_losses:
        df = pd.DataFrame(data={'step': steps, 'loss': losses})
        df.to_csv(train_loss_csv_path, index=False)
        plot_train_loss_graph(steps, losses, train_loss_png_path)


def train_model_with_experience_replay_data_set(
    emulator: Emulator,
    max_taken_actions_per_step: int,
    criterion: CrossEntropyLoss,
    optimizer: AdamW,
    experience_replay_data_set: List[Tuple[ExampleInputTokens, float]],
    context_size: int, batch_size: int, device: torch.device,
    model: DecoderOnlyTransformerModel, total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool,
    max_grad_norm: float, print_model_outputs: bool,
) -> List[Tuple[ExampleInputTokens, float]]:
    """
    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    new_train_examples = generate_examples(
        emulator,
        max_taken_actions_per_step,
        context_size,
        batch_size,
        device,
        model,
        puzzle_train_examples, cell_value_size,
        discount, padding_char)

    experience_replay_data_set_size = 4096
    experience_replay_data_set += new_train_examples
    experience_replay_data_set = trim_list(
        experience_replay_data_set,
        experience_replay_data_set_size,
    )

    min_experience_replay_data_set_size = 4 * batch_size
    loss = None

    if len(experience_replay_data_set) >= min_experience_replay_data_set_size:
        dataset = MyDataset(
            experience_replay_data_set, context_size, num_classes,)

        loss = train(
            criterion,
            optimizer,
            dataset, batch_size, shuffle_train_examples, model,
            max_grad_norm, device,)

        if print_model_outputs:
            print_model_outputs_for_train_examples(
                dataset, batch_size, model, device,)

    return experience_replay_data_set, loss
