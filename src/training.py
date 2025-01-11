import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple
import random
from agent import make_example_tensor, select_action_with_deep_q_network, generate_episode_with_policy, Agent, evaluate_solution
from context import tokens_to_text, prepare_context
from report import plot_train_loss_graph, plot_total_rewards_graph
from model import ActionValueNetworkModel
from environment import Environment, generate_cell_actions
from configuration import Configuration
from q_learning import Experience, unbin_action_value, CellAddress, bin_action_value, trim_list, Cell
from vision import flip_board, rotate_90_clockwise


def get_target_action_value(
        experience: Experience,
        config: Configuration,
        context_size: int,
        cell_value_size: int,
        minimum_action_value: float,
        maximum_action_value: float,
        num_classes: int,
        batch_size: int,
        discount: float,
        device: torch.device,
        target_action_value_network: ActionValueNetworkModel,
        verbose: bool,
) -> float:
    """
    This software used reinforcement learning.
    It uses Q-learning.

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

    best_action, best_action_value, action_values = select_action_with_deep_q_network(
        example_input,
        current_state,
        candidate_actions,
        config,
        context_size,
        batch_size,
        device,
        target_action_value_network,
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
                 agent: Agent,
                 ):
        self.__train_examples = train_examples
        self.__config = config
        self.__device = device
        self.__agent = agent
        self.__printed = False

    def __len__(self):
        return len(self.__train_examples)

    def __getitem__(self, idx):
        experience = self.__train_examples[idx]
        reward = experience.reward()

        # action_index = candidate_action.cell_value()
        action_index = experience.correct_action_index()

        log_probs = experience.log_probs()

        item = (action_index, reward, log_probs,)

        return item


def train_action_value_network(
        criterion: nn.NLLLoss,
        optimizer: AdamW,
        dataset: MyDataset, batch_size: int, shuffle_train_examples: bool, agent: Agent,
        max_grad_norm: float, device: torch.device,
):
    action_value_network = agent.action_value_network()
    action_value_network.train()

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_examples)

    data = next(iter(train_loader))

    optimizer.zero_grad()
    (inputs, targets, action_indices) = data

    inputs = inputs.to(device)
    targets = targets.to(device)

    # outputs has shape [batch_size, num_actions, num_classes].
    # We need a shape of [batch_size, num_classes] to use the criterion.
    all_predicted_action_values = action_value_network(inputs)

    outputs = all_predicted_action_values[torch.arange(
        batch_size), action_indices]

    loss = criterion(outputs, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        action_value_network.parameters(), max_grad_norm)
    optimizer.step()
    loss = loss.cpu().item()

    for t in inputs:
        del t
    del targets
    del outputs

    agent.step_policy_network(
        inputs, all_predicted_action_values.detach(), action_indices)

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


def augment_examples(
        puzzle_train_examples: List[Tuple[List[List[Cell]], List[List[Cell]]]],
) -> List[Tuple[List[List[Cell]], List[List[Cell]]]]:
    augmented_puzzle_train_examples = []
    for i in range(len(puzzle_train_examples)):
        for rotations in range(4):
            for horizontal_flip in range(2):
                for vertical_flip in range(2):
                    augmented_example = generate_training_puzzle_example(
                        puzzle_train_examples,
                        i, rotations, horizontal_flip, vertical_flip,)
                    augmented_puzzle_train_examples.append(augmented_example)
    return augmented_puzzle_train_examples


def train_model_using_experience_replay(
    environment: Environment,
    config: Configuration,
    context_size: int, batch_size: int, device: torch.device,
    agent: Agent,
    puzzle_train_examples: List[Tuple[List[List[Cell]], List[List[Cell]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool, lr: float, weight_decay: float,
    max_grad_norm: float, print_model_outputs: bool, save_step_losses: bool,
    max_episodes: int,
):
    # Augment examples
    augmented_puzzle_train_examples = augment_examples(puzzle_train_examples)
    criterion = None
    optimizer = None

    steps = []
    losses = []

    experience_replay_data_set = []

    step = 0

    winning_streak_length = 0

    while len(environment.recorded_episodes()) < max_episodes:
        if step % config.target_network_update_period == 0:
            agent.update_target_action_value_network()

        old_number_of_episodes = len(environment.recorded_episodes())

        augmented_example_index = len(environment.recorded_episodes()) % len(
            augmented_puzzle_train_examples)

        if augmented_example_index == 0:
            np.random.shuffle(augmented_puzzle_train_examples)

        experience_replay_data_set, percentage_correct_cells, loss = train_model_with_experience_replay_data_set(
            config,
            environment,
            criterion,
            optimizer,
            experience_replay_data_set,
            context_size, batch_size, device,
            agent,
            augmented_puzzle_train_examples[augmented_example_index], cell_value_size,
            discount, padding_char, num_classes, shuffle_train_examples,
            max_grad_norm, print_model_outputs,
        )

        if len(environment.recorded_episodes()) > old_number_of_episodes:
            if percentage_correct_cells >= config.min_percentage_correct_cells:
                winning_streak_length += 1
            else:
                winning_streak_length = 0

        if loss != None:
            episode = len(environment.recorded_episodes())
            steps.append(step)
            losses.append(loss)
            print(
                f"Step: {step}  Episode: {episode}  percentage_correct_cells: {percentage_correct_cells}  winning_streak_length: {winning_streak_length}  loss: {loss:.8f}")

        if winning_streak_length == config.max_winning_streak_length:
            # Early stopping.
            break

        step += 1

    dynamic_time_marker = datetime.now(
        timezone.utc).isoformat().replace(':', '-')
    train_loss_csv_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.csv"
    train_loss_png_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.png"

    if save_step_losses:
        df = pd.DataFrame(data={'step': steps, 'loss': losses})
        df.to_csv(train_loss_csv_path, index=False)
        plot_train_loss_graph(steps, losses, train_loss_png_path)

    # Plot total rewards per episode
    total_rewards_csv_path = f"/workspace/reports/{dynamic_time_marker}-total_rewards.csv"
    total_rewards_png_path = f"/workspace/reports/{dynamic_time_marker}-total_rewards.png"
    total_rewards = environment.get_total_rewards_per_episode()
    episodes = range(len(total_rewards))
    df = pd.DataFrame(data={'episodes': episodes,
                      'total_rewards': total_rewards})
    df.to_csv(total_rewards_csv_path, index=False)
    plot_total_rewards_graph(episodes,
                             total_rewards, total_rewards_png_path)


def generate_training_puzzle_example(
    puzzle_train_examples: List[Tuple[List[List[Cell]], List[List[Cell]]]],
    i: int,
    rotations: int,
    horizontal_flip: int,
    vertical_flip: int,
) -> Tuple[List[List[Cell]], List[List[Cell]]]:

    puzzle_example = puzzle_train_examples[i]

    (raw_example_input, raw_example_output) = puzzle_example

    for _ in range(rotations):
        raw_example_input = rotate_90_clockwise(raw_example_input)
        raw_example_output = rotate_90_clockwise(raw_example_output)

    if horizontal_flip == 0:
        raw_example_input = flip_board(
            raw_example_input, 'horizontal')
        raw_example_output = flip_board(
            raw_example_output, 'horizontal')

    if vertical_flip == 0:
        raw_example_input = flip_board(
            raw_example_input, 'vertical')
        raw_example_output = flip_board(
            raw_example_output, 'vertical')

    return raw_example_input, raw_example_output


def train_model_with_experience_replay_data_set(
    config: Configuration,
    environment: Environment,
    criterion: nn.NLLLoss,
    optimizer: AdamW,
    experience_replay_data_set: List[Experience],
    context_size: int, batch_size: int, device: torch.device,
    agent: Agent,
    puzzle_train_example: Tuple[List[List[Cell]], List[List[Cell]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool,
    max_grad_norm: float, print_model_outputs: bool,
) -> List[Experience]:
    """
    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    example_input, example_output = puzzle_train_example

    # Basically use on-policy data.
    experience_replay_data_set = []

    new_train_examples = generate_episode_with_policy(
        environment,
        config,
        context_size,
        batch_size,
        device,
        agent,
        example_input,
        example_output,
    )

    experience_replay_data_set_size = 4096
    experience_replay_data_set += new_train_examples
    experience_replay_data_set = trim_list(
        experience_replay_data_set,
        experience_replay_data_set_size,
    )

    min_experience_replay_data_set_size = batch_size
    loss = None

    if len(experience_replay_data_set) >= min_experience_replay_data_set_size:
        dataset = MyDataset(
            experience_replay_data_set, config, device, agent,)

        loss = agent.step_policy_network_with_supervision(dataset)

    example_input, current_state = environment.get_observations()
    correct_cells, total_cells = evaluate_solution(
        current_state, example_output)
    percentage_correct_cells = correct_cells / total_cells

    return experience_replay_data_set, percentage_correct_cells, loss
