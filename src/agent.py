import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import numpy as np
import random
import copy
from typing import List, Tuple
from context import tokens_to_text, make_example_tensor, prepare_context
from context import state_to_text, get_puzzle_starting_state
from q_learning import QLearningAction, Cell, Experience, GameState
from model import ActionValueNetworkModel, PolicyNetworkModel
from environment import Environment
from configuration import Configuration
from vision import flip_board, rotate_90_clockwise


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

        (inputs, action_indices) = data

        inputs = [t.to(device) for t in inputs]
        action_indices = action_indices.to(device)

        # Get this quantity:     log π(a|s)
        action_mean_logits = policy_network(inputs)

        batch_size = action_mean_logits.shape[0]

        # L = - log π(a|s)
        log_softmax_output = F.log_softmax(action_mean_logits, dim=-1)

        criterion = nn.NLLLoss()

        loss = criterion(log_softmax_output, action_indices)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_network.parameters(), self.__config.max_grad_norm)
        optimizer.step()
        loss = loss.cpu().item()

        return loss


def apply_puzzle_action_value_policy(puzzle_examples, agent: Agent,
                                     padding_char: str, cell_value_size: int,
                                     context_size: int, batch_size: int,
                                     device,):

    environment = Environment(cell_value_size)
    for example_input, example_output in puzzle_examples:
        print("example")
        solve_puzzle_example_auto_regressive(
            environment,
            example_input, agent,
            padding_char, context_size, batch_size,
            device,)

        example_input, current_state = environment.get_observations()

        print("final output_state")
        print_current_state(example_input, current_state, padding_char)

        # Print the example_target.
        example_output = get_puzzle_starting_state(
            example_output, "example_input")
        print("Expected output")
        print_current_state(
            example_input, example_output, padding_char)

        result = "PASS" if current_state == example_output else "FAIL"
        print(f"Result: {result}")


def solve_puzzle_example_auto_regressive(environment: Environment,
                                         example_input: List[List[int]], agent: Agent, padding_char: str,
                                         context_size: int, batch_size: int,
                                         device: torch.device):
    agent.policy_network().eval()

    environment.set_puzzle_example(example_input, None)

    example_input, current_state = environment.get_observations()

    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(example_input, current_state, padding_char)

    verbose = True

    while not environment.is_in_terminal_state():
        candidate_actions = environment.list_actions()

        example_input, current_state = environment.get_observations()

        best_action_index = select_action_with_policy_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            agent.policy_network(),
            verbose,
        )

        best_action = candidate_actions[best_action_index]

        immediate_reward = environment.take_action(best_action)

        example_input, current_state = environment.get_observations()

        print("AUTO-REGRESSIVE wannabe AGI megabot current state")
        print_current_state(example_input, current_state, padding_char)

    return current_state


def print_current_state(example_input, current_state, padding_char):
    example_input_text = ""
    example_input_text += "exampleInput" + "\n"
    example_input_text += state_to_text(example_input)

    current_state_text = ""
    current_state_text += "currentState" + "\n"
    current_state_text += state_to_text(current_state)

    print(example_input_text)
    print(current_state_text)


def select_action_with_deep_q_network(
        example_input: List[List[Cell]],
        current_state: List[List[Cell]],
        candidate_actions: list[QLearningAction],
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        action_value_network: ActionValueNetworkModel,
        verbose: bool,
) -> Tuple[QLearningAction, int, List[Tuple[int, float]]]:

    # Note that all candidate actions act on the same cell.
    candidate_action = candidate_actions[0]

    input_tokens = prepare_context(
        example_input, current_state, candidate_action, padding_char)

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
        candidate_actions: list[QLearningAction],
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        policy_network: PolicyNetworkModel,
        verbose: bool,
) -> int:
    """
    The policy network outputs a probability distribution over actions.
    Sample an action index using the policy network.
    """
    # Note that all candidate actions act on the same cell.
    candidate_action = candidate_actions[0]

    input_tokens = prepare_context(
        example_input, current_state, candidate_action, padding_char)

    if verbose:
        print("input_text")
        print(tokens_to_text(input_tokens))

    # Add a dimension for the batch_size
    inputs = list(map(lambda tensor: tensor.unsqueeze(0),
                      make_example_tensor(input_tokens, context_size)))

    inputs = [t.to(device) for t in inputs]
    log_probs = policy_network(inputs)
    temperature = 1.0
    probs = torch.exp(log_probs / temperature)
    dist = torch.distributions.Categorical(probs)

    # Sampling fom the probability distribution does the exploration.
    samples = dist.sample()
    best_action_index = samples[0].item()

    return best_action_index


def play_game_using_model(
        environment: Environment,
        max_taken_actions_per_step: int,
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        agent: Agent,
        puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]], cell_value_size: int) -> List[Experience]:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by a random player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.

    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value),
    until the end of the game is reached.

    See:
    Amortized Planning with Large-Scale Transformers: A Case Study on Chess
    https://arxiv.org/abs/2402.04494

    See https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action
    """

    agent.policy_network().eval()
    replay_buffer = []

    if environment.is_in_terminal_state():
        i = random.randrange(0, len(puzzle_train_examples))
        puzzle_example = puzzle_train_examples[i]

        (raw_example_input, raw_example_output) = puzzle_example

        rotations = random.randrange(0, 4)

        for _ in range(rotations):
            raw_example_input = rotate_90_clockwise(raw_example_input)
            raw_example_output = rotate_90_clockwise(raw_example_output)

        if random.randrange(0, 2) == 0:
            raw_example_input = flip_board(
                raw_example_input, 'horizontal')
            raw_example_output = flip_board(
                raw_example_output, 'horizontal')

        if random.randrange(0, 2) == 0:
            raw_example_input = flip_board(
                raw_example_input, 'vertical')
            raw_example_output = flip_board(
                raw_example_output, 'vertical')

        environment.set_puzzle_example(raw_example_input, raw_example_output)

    verbose = False

    while not environment.is_in_terminal_state() and \
            len(replay_buffer) < max_taken_actions_per_step:
        candidate_actions = environment.list_actions()

        example_input, current_state = environment.get_observations()
        current_state = copy.deepcopy(current_state)

        best_action_index = select_action_with_policy_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            agent.policy_network(),
            verbose,
        )

        best_action = candidate_actions[best_action_index]

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
        )
        replay_buffer.append(experience)

    return replay_buffer
