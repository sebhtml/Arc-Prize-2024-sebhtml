import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
from file_storage import FileStorageReader
from agent import make_sample_tensor
from playout_simulation import tokens_to_text


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
    def __init__(self, h5_file_path: str, context_size: int, num_classes: int):
        self.context_size = context_size
        self.num_classes = num_classes
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
        item_input = make_sample_tensor(input_tokens, self.context_size)

        action_value = example[1]
        action_value_bin = bin_action_value(
            action_value, self._minimum_action_value, self._maximum_action_value, self.num_classes)
        action_value_bin = torch.tensor(action_value_bin)
        action_value_bin = F.one_hot(
            action_value_bin, num_classes=self.num_classes).float()

        item = (item_input, action_value_bin)
        return item


def train(dataset: MyDataset, batch_size: int, shuffle_train_examples: bool, step: int, model,
          num_epochs: int, lr: float, weight_decay: float, max_grad_norm: float, device,):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trained_train_examples = dataset.__len__()
    num_steps = num_epochs * math.ceil(trained_train_examples / batch_size)

    print(f"trained_train_examples {trained_train_examples}")
    print(f"batch_size {batch_size}")
    print(f"num_epochs {num_epochs}")
    print(f"num_steps {num_steps}")

    steps = []
    losses = []

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_examples, num_workers=8)

    for epoch in range(num_epochs):
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
            for t in inputs:
                del t
            del targets
            del outputs
            step += 1

    return step, steps, losses


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


def print_model_outputs_for_train_examples(dataset: MyDataset, batch_size: int, model, device,):
    print("[after training] print_model_outputs_for_train_examples")
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
