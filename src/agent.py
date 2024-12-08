import torch
import numpy as np
import random
import copy
from typing import List, Tuple
from context import tokenize_example_input, tokens_to_text, make_example_tensor
from context import state_to_text, QLearningExample
from vision import do_visual_fixation
from q_learning import QLearningAction, Cell, ReplayBuffer, Experience, GameState
from q_learning import sum_of_future_rewards
from model import DecoderOnlyTransformerModel
from emulator import Emulator


def apply_puzzle_action_value_policy(puzzle_examples, model,
                                     padding_char: str, cell_value_size: int,
                                     context_size: int, batch_size: int,
                                     device,):
    emulator = Emulator(cell_value_size)
    for example_input, example_target in puzzle_examples:
        print("example")
        solve_puzzle_example_auto_regressive(
            emulator,
            example_input, model,
            padding_char, context_size, batch_size,
            device,)

        example_input, current_state = emulator.game_state()

        print("final output_state")
        print_current_state(example_input, current_state, padding_char)

        # TODO make the code work to print the example_target.
        # print("Expected output")
        # print_current_state(
        # example_input, example_target, padding_char)


def solve_puzzle_example_auto_regressive(emulator: Emulator,
                                         example_input: List[List[int]], model: DecoderOnlyTransformerModel, padding_char: str,
                                         context_size: int, batch_size: int,
                                         device: torch.device):
    model.eval()

    emulator.set_puzzle_example(example_input, None)

    example_input, current_state = emulator.game_state()

    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(example_input, current_state, padding_char)

    verbose = True

    while not emulator.is_in_terminal_state():
        candidate_actions = emulator.list_actions()

        example_input, current_state = emulator.game_state()

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            model,
            verbose,
        )

        if best_action == None:
            print_current_state(example_input, current_state, padding_char)
            raise Exception("Failed to select action")

        immediate_reward = emulator.take_action(best_action)

        example_input, current_state = emulator.game_state()

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
        model: DecoderOnlyTransformerModel,
        verbose: bool,
):
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
    outputs = model(inputs)

    action_values = []

    # outputs.shape is [batch_size, num_actions, num_classes]
    for action_index in range(outputs.shape[1]):
        action_value = outputs[0, action_index, :].argmax(dim=-1).item()
        action_values.append([action_index, action_value])

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
        emulator: Emulator,
        max_taken_actions_per_step: int,
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]], cell_value_size: int) -> ReplayBuffer:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by a random player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.

    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value),
    until the end of the game is reached.
    """

    model.eval()
    replay_buffer = ReplayBuffer()

    if emulator.is_in_terminal_state():
        i = random.randrange(0, len(puzzle_train_examples))
        puzzle_example = puzzle_train_examples[i]

        (raw_example_input, raw_example_output) = puzzle_example
        emulator.set_puzzle_example(raw_example_input, raw_example_output)

    verbose = False

    while not emulator.is_in_terminal_state() and \
            len(replay_buffer.experiences()) < max_taken_actions_per_step:
        candidate_actions = emulator.list_actions()

        example_input, current_state = emulator.game_state()
        current_state = copy.deepcopy(current_state)

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            model,
            verbose,
        )

        immediate_reward = emulator.take_action(best_action)

        example_input, next_state = emulator.game_state()
        next_state = copy.deepcopy(next_state)

        experience = Experience(
            GameState(example_input, current_state),
            best_action,
            immediate_reward,
            GameState(example_input, next_state),
        )
        replay_buffer.add_experience(experience)

    return replay_buffer


def extract_action_examples(replay_buffer: ReplayBuffer, discount: float, padding_char: str) -> List[QLearningExample]:
    """
    Generate (state, action, action_value) action examples for a puzzle example.
    """

    examples = []

    experiences = replay_buffer.experiences()
    for idx, experience in enumerate(experiences):
        reward = experience.reward()
        current_state = experience.state().current_state()
        candidate_action = experience.action()

        expected_rewards = sum_of_future_rewards(
            reward, discount, current_state, candidate_action)
        action_value = expected_rewards
        is_terminal = idx == len(experiences)
        example = QLearningExample(
            experience, action_value, is_terminal)

        examples.append(example)

    return examples


def generate_examples(
        emulator: Emulator,
        max_taken_actions_per_step: int,
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]], cell_value_size: int,
        discount: float, padding_char: str
) -> List[QLearningExample]:
    """
    Generate training examples from a puzzle example.
    """

    replay_buffer = play_game_using_model(
        emulator,
        max_taken_actions_per_step,
        padding_char,
        context_size,
        batch_size,
        device,
        model,
        puzzle_train_examples, cell_value_size)

    action_examples = extract_action_examples(
        replay_buffer, discount, padding_char)

    return action_examples
