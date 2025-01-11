import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
import sys
import copy
from typing import List, Tuple
from context import tokens_to_text, make_example_tensor, prepare_context
from context import state_to_text, get_puzzle_starting_state
from q_learning import QLearningAction, Cell, CellAddress, Experience, GameState, trim_list
from model import ActionValueNetworkModel, PolicyNetworkModel
from environment import Environment
from configuration import Configuration
from episode_renderer import print_game_state_with_colors


class Agent:
    def __init__(self, config: Configuration, device: torch.device):
        # Device and config
        self.__device = device
        self.__config = config

        # Action value network for DQN
        self.__action_value_network = None

        if self.__config.use_action_value_network:
            self.__action_value_network = ActionValueNetworkModel(
                config, device)
            self.__action_value_network.to(device)

        # Target action value network for DQN
        self.__target_action_value_network = None

        # Policy network for policy gradient method
        if self.__config.use_policy_network:
            self.__policy_network = PolicyNetworkModel(config, device)
            self.__policy_network.to(device)
            self.__policy_network.train()
            self.__policy_network_optimizer = AdamW(self.__policy_network.parameters(),
                                                    lr=config.lr, weight_decay=config.weight_decay)
        else:
            self.__policy_network = None
            self.__policy_network_optimizer = None

    def action_value_network(self) -> ActionValueNetworkModel:
        return self.__action_value_network

    def target_action_value_network(self) -> ActionValueNetworkModel:
        return self.__target_action_value_network

    def policy_network(self) -> PolicyNetworkModel:
        return self.__policy_network

    def update_target_action_value_network(self):
        if self.action_value_network() == None:
            return
        if self.__target_action_value_network != None:
            self.__target_action_value_network.to('cpu')
            del self.__target_action_value_network
            self.__target_action_value_network = None
        self.__target_action_value_network = copy.deepcopy(
            self.action_value_network())
        self.__target_action_value_network.to(
            self.__device)

    def step_policy_network(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        all_predicted_action_values: torch.Tensor,
        action_indices: torch.Tensor,
    ):
        """
        L_actor = - log π(a|s) * A(s, a)

        A(s, a) = Q(s, a) - V(s)

        V(s) = max_a' { Q(s, a') }

        See
        https://en.wikipedia.org/wiki/Temporal_difference_learning

        See
        Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.

        See
        Learning to Predict by the Methods of Temporal Differences.
        https://link.springer.com/article/10.1007/BF00115009

        See 
        Off-Policy Actor-Critic
        https://icml.cc/2012/papers/268.pdf

        See Actor-Critic Algorithms
        https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
        """

        if not self.__config.use_policy_network:
            return

        policy_network = self.policy_network()
        optimizer = self.__policy_network_optimizer

        # Get this quantity:     log π(a|s)
        all_predicted_action_probabilities = policy_network(inputs)

        batch_size = all_predicted_action_probabilities.shape[0]
        action_probabilities = all_predicted_action_probabilities[torch.arange(
            batch_size), action_indices]

        # Q(s, a)
        action_values = all_predicted_action_values[torch.arange(
            batch_size), action_indices].argmax(dim=-1).float()
        print(f"action_values {action_values}")

        # V(s) = max_a' { Q(s, a') }
        max_action_values, _ = all_predicted_action_values.argmax(
            dim=-1).max(dim=-1)
        print(f"max_action_values {max_action_values}")

        # A(s, a) = Q(s, a) - V(s)
        advantages = action_values - max_action_values

        probabilities = torch.exp(action_probabilities)
        probabilities = probabilities / \
            torch.sum(probabilities, dim=-1, keepdim=True)

        print(f"action_probabilities {probabilities}")
        print(f"advantages {advantages}")

        # loss
        # L_actor = - log π(a|s) * A(s, a)
        loss = - action_probabilities * advantages
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_network.parameters(), self.__config.max_grad_norm)
        optimizer.step()
        loss = loss.cpu().item()

        print(f"policy network loss {loss}")

    def step_policy_network_with_supervision(
        self,
        dataset,
    ):
        """
        L_actor = - log π(a|s)
        """

        device = self.__device

        train_loader = DataLoader(
            dataset, batch_size=self.__config.batch_size, shuffle=True)

        data = next(iter(train_loader))

        policy_network = self.__policy_network
        optimizer = self.__policy_network_optimizer

        (action_indices, rewards, log_probs,) = data

        action_indices = action_indices.to(device)

        batch_size = log_probs.shape[0]

        log_probs = log_probs[torch.arange(
            batch_size), action_indices]

        # Reinforce
        # L = -log P(a | s) * v
        # loss = -torch.mean(log_probs * rewards)

        # Negative log likelihood.
        # L = -log P(a | s)
        loss = -log_probs
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_network.parameters(), self.__config.max_grad_norm)
        optimizer.step()
        loss = loss.cpu().item()

        return loss


def flatten(xss):
    return [x for xs in xss for x in xs]


def evaluate_solution(actual: List[List[Cell]], expected: List[List[Cell]]) -> Tuple[int, int]:
    """
    Return the number of correct cells and total cells
    """
    actual = flatten(actual)
    expected = flatten(expected)

    if len(actual) != len(expected):
        raise Exception("Invalid comparison")

    correct = len(
        list(filter(lambda x: x[1] == expected[x[0]], enumerate(actual))))
    return correct, len(expected)


def test_policy(puzzle_examples, agent: Agent,
                config: Configuration, cell_value_size: int,
                context_size: int, batch_size: int,
                device: torch.device,
                environment: Environment,):

    for example_input, example_output in puzzle_examples:
        print("example")
        generate_episode_with_policy(
            environment,
            config,
            context_size,
            batch_size,
            device,
            agent,
            example_input,
            None,
        )

        episode = environment.recorded_episodes()[-1]
        for state in episode:
            example_input = state.example_input()
            current_state = state.current_state()
            print_current_state(
                example_input, current_state, config.padding_char,)

        example_input, current_state = environment.get_observations()

        # Print the example_target.
        print("Expected output")
        print_current_state(
            example_input, example_output, config.padding_char,)

        correct_cells, total_cells = evaluate_solution(
            current_state, example_output)
        percentage_correct_cells = correct_cells / total_cells
        result = "PASS" if correct_cells == total_cells else "FAIL"
        print(
            f"Result:  correct_cells: {correct_cells}  total_cells: {total_cells}  percentage_correct_cells: {percentage_correct_cells}  result: {result}")


def print_current_state(example_input, current_state, padding_char):
    game_state = GameState(example_input, current_state)
    print_game_state_with_colors(game_state, sys.stdout)


def select_action_with_deep_q_network(
        example_input: List[List[Cell]],
        current_state: List[List[Cell]],
        candidate_actions: list[QLearningAction],
        config: Configuration,
        context_size: int,
        batch_size: int,
        device: torch.device,
        action_value_network: ActionValueNetworkModel,
        verbose: bool,
) -> Tuple[QLearningAction, int, List[Tuple[int, float]]]:

    # Note that all candidate actions act on the same cell.
    candidate_action = candidate_actions[0]
    cell_address = CellAddress(candidate_action.row(), candidate_action.col(),)

    input_tokens = prepare_context(
        example_input, cell_address, config.padding_char,
        config.num_visual_fixations,
        config.visual_fixation_height,
        config.visual_fixation_width,
    )

    if verbose:
        print("input_text")
        print(tokens_to_text(input_tokens))

    # Add a dimension for the batch_size
    inputs = list(map(lambda tensor: tensor.unsqueeze(0),
                      make_example_tensor(input_tokens, context_size)))

    inputs = [t.to(device) for t in inputs]
    log_softmax_outputs = action_value_network(inputs)

    action_values = []

    # outputs.shape is [batch_size, num_actions, num_classes]

    # Use a distributional mean action value.
    #
    # See
    # Rainbow: Combining Improvements in Deep Reinforcement Learning
    # https://arxiv.org/pdf/1710.02298
    #
    # See
    # A Distributional Perspective on Reinforcement Learning
    # https://arxiv.org/abs/1707.06887
    #
    # Evaluates the Einstein summation convention on the operands.

    outputs = None
    use_mean_action_value = False

    if use_mean_action_value:
        num_classes = log_softmax_outputs.shape[-1]
        atoms = torch.arange(
            num_classes, device=log_softmax_outputs.device).float()

        probability_outputs = torch.exp(log_softmax_outputs)
        outputs = torch.einsum('n,ban->ba', atoms, probability_outputs)
    else:
        outputs = log_softmax_outputs.argmax(-1).float()

    for action_index in range(outputs.shape[1]):
        mean_action_value = outputs[0, action_index].item()
        action_values.append([action_index, mean_action_value])

    np.random.shuffle(action_values)

    best_action = None
    best_action_value = None

    for action_index, action_value in action_values:
        candidate_action = candidate_actions[action_index]
        row = candidate_action.row()
        col = candidate_action.col()
        cell_value = candidate_action.cell_value()

        if verbose:
            print(
                f"Testing action  row: {row}  col: {col}  cell_value: {cell_value} action_value: {action_value}")
        if best_action_value == None or action_value > best_action_value:
            best_action = candidate_action
            best_action_value = action_value

    for t in inputs:
        del t
    del outputs

    return best_action, best_action_value, action_values


def select_action_with_policy_network(
        example_input: List[List[Cell]],
        current_state: List[List[Cell]],
        cell_addresses: List[CellAddress],
        config: Configuration,
        context_size: int,
        batch_size: int,
        device: torch.device,
        policy_network: PolicyNetworkModel,
        verbose: bool,
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    The policy network outputs a probability distribution over actions.
    Sample an action index using the policy network.
    """

    inputs = []

    for cell_address in cell_addresses:
        input_tokens = prepare_context(
            example_input, cell_address, config.padding_char,
            config.num_visual_fixations,
            config.visual_fixation_height,
            config.visual_fixation_width,
        )

        if verbose:
            print("input_text")
            print(tokens_to_text(input_tokens))

        # Add a dimension for the batch_size
        inputs.append(make_example_tensor(input_tokens, context_size))

    inputs = torch.stack(inputs)

    inputs = inputs.to(device)
    logits = policy_network(inputs)
    temperature = 1.0
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    # Sampling fom the probability distribution does the exploration.
    samples = dist.sample()
    best_action_indexes = samples.tolist()

    log_probs = F.log_softmax(logits, dim=-1)

    return best_action_indexes, log_probs


def generate_episode_with_policy(
        environment: Environment,
        config: Configuration,
        context_size: int,
        batch_size: int,
        device: torch.device,
        agent: Agent,
        example_input: List[List[Cell]],
        example_output: List[List[Cell]],
) -> List[Experience]:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by the policy network.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.

    See:
    Amortized Planning with Large-Scale Transformers: A Case Study on Chess
    https://arxiv.org/abs/2402.04494

    See https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action
    """

    replay_buffer = []

    environment.set_puzzle_example(example_input, example_output)

    verbose = False

    # Play a full episode.
    while not environment.is_in_terminal_state():

        candidate_actions = environment.list_actions()
        cell_addresses = []
        for candidate_action in candidate_actions:
            if candidate_action.cell_value() == 0:
                cell_address = CellAddress(
                    candidate_action.row(), candidate_action.col(),)
                cell_addresses.append(cell_address)

        np.random.shuffle(cell_addresses)
        cell_addresses = trim_list(cell_addresses, batch_size)

        example_input, current_state = environment.get_observations()
        current_state = copy.deepcopy(current_state)

        best_action_indexes, log_probs = select_action_with_policy_network(
            example_input,
            current_state,
            cell_addresses,
            config,
            context_size,
            batch_size,
            device,
            agent.policy_network(),
            verbose,
        )

        for i in range(len(cell_addresses)):
            cell_address = cell_addresses[i]
            best_action_index = best_action_indexes[i]
            cell_log_probs = log_probs[i]
            row = cell_address.row()
            col = cell_address.col()
            cell_value = best_action_index
            best_action = QLearningAction(row, col, cell_value,)

            immediate_reward = environment.take_action(best_action)
            expected_cell_value = environment.get_correct_action(
                best_action.row(), best_action.col())

            example_input, next_state = environment.get_observations()
            next_state = copy.deepcopy(next_state)

            experience = Experience(
                GameState(example_input, current_state),
                best_action,
                immediate_reward,
                GameState(example_input, next_state),
                expected_cell_value,
                cell_log_probs,
            )
            replay_buffer.append(experience)

    return replay_buffer
