import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import copy
from datetime import datetime, timezone
from typing import List, Tuple
from agent import make_example_tensor, generate_examples, select_action_with_deep_q_network
from context import tokens_to_text, tokenize_example_input
from report import plot_train_loss_graph
from model import DecoderOnlyTransformerModel
from emulator import Emulator, generate_cell_actions
from vision import do_visual_fixation
from configuration import Configuration
from q_learning import Experience


def unbin_action_value(action_value_bin: int, minimum_action_value: float, maximum_action_value: float, num_classes: int) -> float:
    """
    Convert a bin between 0 and num_classes-1 to a float between minimum_action_value and maximum_action_value
    """
    minimum_action_value_bin = 0
    maximum_action_value_bin = num_classes - 1

    action_value = minimum_action_value + (
        ((action_value_bin - minimum_action_value_bin) / (maximum_action_value_bin - minimum_action_value_bin)) * (maximum_action_value - minimum_action_value))
    return max(min(action_value, maximum_action_value), minimum_action_value)


def bin_action_value(action_value: float, minimum_action_value: float, maximum_action_value: float, num_classes: int) -> int:
    """
    convert action_value to { 0, 1, ..., num_classes - 1 }
    Example:
    action_value: 3.0
    _minimum_action_value: -4.0
    _maximum_action_value: 7.0
    action_value_bin = (3.0 - -4.0) / (7.0 - -4.0)
    """
    minimum_action_value_bin = 0
    maximum_action_value_bin = num_classes - 1

    action_value_bin = minimum_action_value_bin + math.floor(
        ((action_value - minimum_action_value) / (maximum_action_value - minimum_action_value)) * (maximum_action_value_bin - minimum_action_value_bin))
    return max(min(action_value_bin, maximum_action_value_bin), minimum_action_value_bin)


def get_target_action_value(
        experience: Experience,
        padding_char: str,
        context_size: int,
        cell_value_size: int,
        minimum_action_value: float,
        maximum_action_value: float,
        num_classes: int,
        batch_size: int,
        discount: float,
        device: torch.device,
        target_model: DecoderOnlyTransformerModel,
        verbose: bool,
) -> float:
    """
    See https://en.wikipedia.org/wiki/Q-learning
    See https://en.wikipedia.org/wiki/Bellman_equation

    See
    Dynamic Programming
    https://www.science.org/doi/10.1126/science.153.3731.34

    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    reward = experience.reward()

    example_input = experience.next_state().example_input()
    current_state = experience.next_state().current_state()

    candidate_actions = generate_cell_actions(
        current_state, cell_value_size)

    is_terminal = len(candidate_actions) == 0

    if is_terminal:
        return reward

    best_action, best_action_value = select_action_with_deep_q_network(
        example_input,
        current_state,
        candidate_actions,
        padding_char,
        context_size,
        batch_size,
        device,
        target_model,
        verbose,
    )

    best_action_value = unbin_action_value(
        best_action_value, minimum_action_value, maximum_action_value, num_classes)

    # We use the Bellman equation.
    # See https://en.wikipedia.org/wiki/Bellman_equation
    #
    # Given an experience
    # (x, a, r, x')
    # where
    #   x                        is the state
    #   a in Gamma(x)            is the action
    #   r = F(s, a)              is the reward
    #   x' = T(s, a)             is the new state
    #
    # We have
    #
    # V(x) = max_{a \in \Gamma(x)} [ F(x, a) + \beta * V(T(x, a)) ]
    #
    # Here, concretely, we have:
    #
    #
    #  Mathematical expression           Python expression
    #  -----------------------------------------------------
    #  a                                 experience.action()
    #  F(x, a)                           reward
    #  \beta                             discount
    #  V(T(x, a))                        best_action_value
    #

    # In Q-learning (https://en.wikipedia.org/wiki/Q-learning)
    # we write the Bellman equation as the following mathematical expresssion:
    #
    # We note the state with s instead of x.
    #
    # The discount is noted with gamma instead of beta.
    #
    # Q: S x A -> R
    #
    #
    # \hat{Q}(s, a) = r + gamma * max_{a' \in \Gamma(s')} \hat{Q}(s', a')
    #
    #
    # with
    # s' = T(s, a)
    # s \in S
    # a \in A
    # a \in \Gamma(s)

    return reward + discount * best_action_value


class MyDataset(Dataset):
    def __init__(self,
                 train_examples: List[Experience],
                 config: Configuration,
                 device: torch.device,
                 target_model: DecoderOnlyTransformerModel,
                 ):
        self.__train_examples = train_examples
        self.__config = config
        self.__device = device
        self.__target_model = target_model

    def __len__(self):
        return len(self.__train_examples)

    def __getitem__(self, idx):
        experience = self.__train_examples[idx]
        example_input = experience.state().example_input()
        current_state = experience.state().current_state()
        candidate_action = experience.action()

        (attented_example_input, attented_current_state, attented_candidate_action,
         translation_x, translation_y) = do_visual_fixation(
            example_input, current_state, candidate_action)

        input_tokens = tokenize_example_input(
            current_state,
            attented_example_input, attented_current_state, self.__config.padding_char)

        item_input = make_example_tensor(
            input_tokens, self.__config.context_size)

        action_index = torch.tensor(experience.action().cell_value())

        verbose = self.__config.verbose_target_network
        action_value = get_target_action_value(
            experience,
            self.__config.padding_char,
            self.__config.context_size,
            self.__config.cell_value_size,
            self.__config.minimum_action_value,
            self.__config.maximum_action_value,
            self.__config.num_classes,
            self.__config.batch_size,
            self.__config.discount,
            self.__device,
            self.__target_model,
            verbose,
        )
        action_value_bin = bin_action_value(
            action_value, self.__config.minimum_action_value, self.__config.maximum_action_value, self.__config.num_classes)
        action_value_bin = torch.tensor(action_value_bin)

        item = (item_input, action_value_bin, action_index)
        return item

    def get_action_value_min_max(self, train_examples: List[Experience]) -> Tuple[float, float]:
        min_action_value = min(map(lambda example: example[1], train_examples))
        max_action_value = max(map(lambda example: example[1], train_examples))
        return min_action_value, max_action_value


def trim_list(lst, k):
    """
    keep at most k elements from list lst
    """
    return lst[-k:] if len(lst) > k else lst


def train(
        criterion: nn.NLLLoss,
        optimizer: AdamW,
        dataset: MyDataset, batch_size: int, shuffle_train_examples: bool, model: DecoderOnlyTransformerModel,
        max_grad_norm: float, device: torch.device,
):
    model.train()

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_examples)

    data = next(iter(train_loader))

    optimizer.zero_grad()
    (inputs, targets, action_indices) = data

    inputs = [t.to(device) for t in inputs]
    targets = targets.to(device)

    # outputs has shape [batch_size, num_actions, num_classes].
    # We need a shape of [batch_size, num_classes] to use the criterion.
    outputs = model(inputs)
    outputs = outputs[torch.arange(batch_size), action_indices]

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


def print_model_outputs_for_train_examples(dataset: MyDataset, batch_size: int, model: DecoderOnlyTransformerModel, device: torch.device,):
    print("[after training] print_model_outputs_for_train_examples")
    inference_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    for data in inference_loader:
        (inputs, targets, action_indices) = data
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs[torch.arange(batch_size), action_indices]
        for idx in range(len(inputs[0])):
            print("--------------------")
            print(f"idx: {idx} ")
            input = [inputs[0][idx], inputs[1][idx],
                     inputs[2][idx]]
            target = targets[idx].item()
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
    config: Configuration,
    context_size: int, batch_size: int, device: torch.device,
    model: DecoderOnlyTransformerModel, total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool, lr: float, weight_decay: float,
    max_grad_norm: float, print_model_outputs: bool, save_step_losses: bool,
    num_steps: int,
):
    criterion = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    emulator = Emulator(cell_value_size)
    steps = []
    losses = []

    target_model = None

    experience_replay_data_set = []
    for step in range(num_steps):
        if step % config.target_network_update_period == 0:
            if target_model != None:
                target_model.to('cpu')
                del target_model
                target_model = None
            target_model = copy.deepcopy(model)
            target_model.to(device)

        experience_replay_data_set, loss = train_model_with_experience_replay_data_set(
            config,
            emulator,
            criterion,
            optimizer,
            experience_replay_data_set,
            context_size, batch_size, device,
            model,
            target_model,
            total_train_examples,
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
    config: Configuration,
    emulator: Emulator,
    criterion: nn.NLLLoss,
    optimizer: AdamW,
    experience_replay_data_set: List[Experience],
    context_size: int, batch_size: int, device: torch.device,
    model: DecoderOnlyTransformerModel,
    target_model: DecoderOnlyTransformerModel,
    total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool,
    max_grad_norm: float, print_model_outputs: bool,
) -> List[Experience]:
    """
    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    new_train_examples = generate_examples(
        emulator,
        config.max_taken_actions_per_step,
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
            experience_replay_data_set, config, device, target_model,)

        loss = train(
            criterion,
            optimizer,
            dataset, batch_size, shuffle_train_examples, model,
            max_grad_norm, device,)

        if print_model_outputs:
            print_model_outputs_for_train_examples(
                dataset, batch_size, model, device,)

    return experience_replay_data_set, loss
