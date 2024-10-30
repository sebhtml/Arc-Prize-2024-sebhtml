from file_storage import FileStorageWriter, SampleInputTokens
import random
import copy
import numpy as np
from typing import List


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


class Cell:
    def __init__(self, value):
        self.__value = value
        self.__changes = 0

    def value(self) -> int:
        return self.__value

    def changes(self) -> int:
        return self.__changes

    def set_value(self, value):
        self.__value = value
        self.__changes += 1


def generate_samples(train_dataset_path: str, total_train_samples: int, puzzle_train_examples, cell_value_size: int,
                     input_gen_mode: str, current_gen_mode: str, discount: float, padding_char: str, correct_color_probability: float):
    writer = FileStorageWriter(train_dataset_path)
    generated_samples = 0
    while generated_samples < total_train_samples:
        samples = []
        for _ in range(1):
            samples += generate_train_action_examples(
                puzzle_train_examples, cell_value_size, input_gen_mode, current_gen_mode, discount, padding_char, correct_color_probability)
        generated_samples += len(samples)
        writer.append(samples)

# TODO this function should receive just one puzzle example


def generate_train_action_examples(puzzle_examples, cell_value_size, input_gen_mode: str, current_gen_mode: str, discount: float, padding_char: str,
                                   correct_color_probability: float):
    """
    Generate (state, action, action_value) experience samples for all puzzle examples.
    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value)
    until the end of the game is reached.
    This is essentially a wrong incorrect half-baked MCTS (Monte Carlo tree search) in the sense that it's a
    tree search. And there is no Monte Carlo or Markov Chain involved here since we are lazy at best.
    """
    train_examples = []
    for puzzle_example in puzzle_examples:
        action_examples = generate_action_examples(
            puzzle_example, cell_value_size, input_gen_mode, current_gen_mode, discount, padding_char, correct_color_probability)
        train_examples += action_examples
    return train_examples


def actions_act_on_same_cell(action_1: QLearningAction, action_2: QLearningAction) -> bool:
    return (action_1.row(), action_1.col()) == (action_2.row(), action_2.col())


def generate_action_examples(puzzle_example, cell_value_size, input_gen_mode: str, current_gen_mode: str, discount: float,
                             padding_char: str, correct_color_probability: float):
    (example_input, example_output) = puzzle_example
    action_examples = []
    example_input = get_starting_current_state(
        example_input, cell_value_size, input_gen_mode)
    current_state = get_starting_current_state(
        example_output, cell_value_size, current_gen_mode)

    assert current_state != None

    candidate_actions = generate_cell_actions(
        current_state, cell_value_size)

    while current_state != example_output:
        best_next_state = None
        best_action = None
        best_action_example = None

        if len(candidate_actions) == 0:
            break

        np.random.shuffle(candidate_actions)

        # Each time that the player assign a color to a cell, the assigned color is either correct or incorrect.
        use_correct_color = True if random.uniform(
            0, 1) <= correct_color_probability else False

        for candidate_action in candidate_actions:
            next_state = copy.deepcopy(current_state)
            row = candidate_action.row()
            col = candidate_action.col()
            cell_value = candidate_action.cell_value()
            next_state[row][col].set_value(cell_value)
            input_text = tokenize_sample_input(
                example_input, current_state, candidate_action, padding_char)
            # Use Q*(s, a) for the action-value.
            action_value = get_q_star_action_value(
                current_state, candidate_action, example_output, discount)
            example = (input_text, action_value)

            cell_color_is_correct: bool = example_output[row][col] == cell_value

            satisfaction: bool = use_correct_color == cell_color_is_correct

            if satisfaction:
                best_next_state = next_state
                best_action = candidate_action
                best_action_example = example

                # We can break as soon as we found an action that assign a correct color or an incorrect color.
                # The only thing that matters is that we shuffled the legal actions before testing the actions.
                break

        # Remove all actions acting on that cell for now.
        candidate_actions = list(
            filter(lambda action: not actions_act_on_same_cell(action, best_action), candidate_actions))

        assert best_next_state != None
        action_examples.append(best_action_example)
        current_state = best_next_state
        assert current_state != None

    return action_examples


def get_starting_current_state(state, cell_value_size: int, mode: str) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = generate_initial_cell_value(
                current_state, row, col, mode, cell_value_size)
            current_state[row][col] = Cell(value)
    return current_state


def generate_initial_cell_value(state, row, col, mode, cell_value_size: int) -> int:
    if mode == "randomize":
        return random.randrange(0, cell_value_size)
    elif mode == "identity":
        return state[row][col]
    elif mode == "zero":
        return 0


def reward(expected_cell_value, cell_value) -> int:
    if expected_cell_value == cell_value:
        return 1.0
    else:
        return -1.0


def get_q_star_action_value(state, action: QLearningAction, example_output, discount) -> int:
    """
    - discount is gamma
    - Q*(s, a) = gamma^0 * r_{t+1} + gamma^1* r_{t+1} + gamma^2 * r_{t+2} + ...
    """
    # Immediate reward is not discounted.
    immediate_reward = reward(
        example_output[action.row()][action.col()], action.cell_value())
    # Discounted future rewards
    maximum_sum_of_discounted_future_rewards = 0.0
    t = 1

    for row in range(len(state)):
        for col in range(len(state[row])):
            # Skip cell because it was already counted as the immediate reward.
            if row == action.row() and col == action.col():
                continue
            # A cell can only be changed once.
            if state[row][col].changes() == 1:
                continue
            # Maximize future expected discounted rewards.
            # Assume perfect play in the future.
            future_reward = reward(
                example_output[row][col], example_output[row][col])
            discounted_reward = discount**t * future_reward
            maximum_sum_of_discounted_future_rewards += discounted_reward
            t += 1

    action_value = immediate_reward + maximum_sum_of_discounted_future_rewards
    return action_value


def generate_cell_actions(current_state, cell_value_size) -> list[QLearningAction]:
    candidate_actions = []
    assert current_state != None
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].changes() > 0:
                continue
            for new_value in range(cell_value_size):
                action = QLearningAction(row, col, new_value)
                candidate_actions.append(action)
    return candidate_actions


def tokenize_sample_input(input_state, current_state, action: QLearningAction, padding_char: str) -> SampleInputTokens:
    """
    Tokenize a sample input for the Q-network Q(s, a).
    Note that:
    - s contains the state which is (input_state, current_state)
    - a contains the action which is (next_state)
    """

    # state s
    input_state_text, full_move_counter, current_state_text = get_state_texts(
        input_state, current_state, padding_char)

    action_text = ""
    action_text += "act" + "\n"
    action_text += action_to_text(current_state, action)

    return SampleInputTokens(
        text_to_tokens(input_state_text),
        text_to_tokens(full_move_counter),
        text_to_tokens(current_state_text),
        text_to_tokens(action_text)
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def get_state_texts(input_state, current_state, padding_char: str):
    input_state_text = ""
    input_state_text += "ini" + "\n"
    input_state_text += input_state_to_text(input_state)

    full_move_counter = ""
    full_move_counter += "cnt" + "\n"
    full_move_counter += get_full_move_counter(current_state, padding_char)

    current_state_text = ""
    current_state_text += "cur" + "\n"
    current_state_text += current_state_to_text(current_state)

    return input_state_text, full_move_counter, current_state_text


def input_state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = state[row][col].value()
            output += str(value)
        output += "\n"
    return output


def get_full_move_counter(state, padding_char: str) -> str:
    """
    Get the full move counter since the beginning of the playout
    """
    full_move_counter: int = 0
    for row in range(len(state)):
        for col in range(len(state[row])):
            changes = state[row][col].changes()
            if changes > 1:
                raise Exception(
                    f"Current state has a cell with {changes}, which is illegal because it is greater than 1.")
            full_move_counter += changes

    output: str = str(full_move_counter).rjust(2, padding_char)
    output += "\n"
    return output


def current_state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if state[row][col].changes() == 0:
                value = '_'
            else:
                value = state[row][col].value()
            output += str(value)
        output += "\n"
    return output


def action_to_text(state, action: QLearningAction) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if row == action.row() and col == action.col():
                value = action.cell_value()
            else:
                value = '_'
            output += str(value)
        output += "\n"
    return output


def compute_action_token(action: QLearningAction, puzzle_height: int, cell_value_size: int) -> int:
    # action a
    # For example, a puzzle with a grid 7x7 has 7*7*10 possible actions.
    # Points (a, b, c) are in a coordinate system of size (WIDTH, HEIGHT, VALUES)
    action_token = action.col() * puzzle_height * cell_value_size + \
        action.row() * cell_value_size + action.cell_value()
    return action_token
