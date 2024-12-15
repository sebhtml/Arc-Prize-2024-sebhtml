import torch
import numpy as np
import random
import copy
from typing import List, Tuple
from context import tokenize_example_input, tokens_to_text, make_example_tensor
from context import state_to_text
from vision import do_visual_fixation
from q_learning import QLearningAction, Cell, Experience, GameState
from model import ActionValueNetworkModel
from environment import Environment
from configuration import Configuration


class Agent:
    def __init__(self, config: Configuration, device: torch.device):
        self.__action_value_network = ActionValueNetworkModel(config, device)
        self.__action_value_network.to(device)

    def action_value_network(self) -> ActionValueNetworkModel:
        return self.__action_value_network


def apply_puzzle_action_value_policy(puzzle_examples, agent: Agent,
                                     padding_char: str, cell_value_size: int,
                                     context_size: int, batch_size: int,
                                     device,):

    environment = Environment(cell_value_size)
    for example_input, example_target in puzzle_examples:
        print("example")
        solve_puzzle_example_auto_regressive(
            environment,
            example_input, agent,
            padding_char, context_size, batch_size,
            device,)

        example_input, current_state = environment.get_observations()

        print("final output_state")
        print_current_state(example_input, current_state, padding_char)

        # TODO make the code work to print the example_target.
        # print("Expected output")
        # print_current_state(
        # example_input, example_target, padding_char)


def solve_puzzle_example_auto_regressive(environment: Environment,
                                         example_input: List[List[int]], agent: Agent, padding_char: str,
                                         context_size: int, batch_size: int,
                                         device: torch.device):
    agent.action_value_network().eval()

    environment.set_puzzle_example(example_input, None)

    example_input, current_state = environment.get_observations()

    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(example_input, current_state, padding_char)

    verbose = True

    while not environment.is_in_terminal_state():
        candidate_actions = environment.list_actions()

        example_input, current_state = environment.get_observations()

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            agent.action_value_network(),
            verbose,
        )

        if best_action == None:
            print_current_state(example_input, current_state, padding_char)
            raise Exception("Failed to select action")

        immediate_reward = environment.take_action(best_action)

        example_input, current_state = environment.get_observations()

        print(f"best_next_state with {best_action_value}")
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
) -> Tuple[QLearningAction, int]:

    # Note that all candidate actions act on the same cell.
    candidate_action = candidate_actions[0]

    (attented_example_input, attented_current_state, attented_candidate_action,
     translation_x, translation_y) = do_visual_fixation(example_input, current_state, candidate_action)

    input_tokens = tokenize_example_input(
        current_state,
        attented_example_input, attented_current_state, padding_char)

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
    num_classes = log_softmax_outputs.shape[-1]
    atoms = torch.arange(
        num_classes, device=log_softmax_outputs.device).float()

    probability_outputs = torch.exp(log_softmax_outputs)
    outputs = torch.einsum('n,ban->ba', atoms, probability_outputs)

    for action_index in range(outputs.shape[1]):
        # argmax_action_value = log_softmax_outputs[0, action_index, :].argmax(dim=-1).item()
        mean_action_value = outputs[0, action_index].item()
        # print(f"argmax_action_value {argmax_action_value}")
        # print(f"mean_action_value {mean_action_value}")
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

    return best_action, best_action_value


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

    agent.action_value_network().eval()
    replay_buffer = []

    if environment.is_in_terminal_state():
        i = random.randrange(0, len(puzzle_train_examples))
        puzzle_example = puzzle_train_examples[i]

        (raw_example_input, raw_example_output) = puzzle_example
        environment.set_puzzle_example(raw_example_input, raw_example_output)

    verbose = False

    while not environment.is_in_terminal_state() and \
            len(replay_buffer) < max_taken_actions_per_step:
        candidate_actions = environment.list_actions()

        example_input, current_state = environment.get_observations()
        current_state = copy.deepcopy(current_state)

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            agent.action_value_network(),
            verbose,
        )

        immediate_reward = environment.take_action(best_action)

        example_input, next_state = environment.get_observations()
        next_state = copy.deepcopy(next_state)

        experience = Experience(
            GameState(example_input, current_state),
            best_action,
            immediate_reward,
            GameState(example_input, next_state),
        )
        replay_buffer.append(experience)

    return replay_buffer
