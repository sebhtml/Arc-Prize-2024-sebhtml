from file_storage import FileStorageWriter, SampleInputTokens
import random
import copy
import numpy as np
from typing import List
import concurrent.futures
import concurrent


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
                     discount: float, padding_char: str, cpu_count: int):
    writer = FileStorageWriter(train_dataset_path)
    generated_samples = 0
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count)
    must_generate_more_samples = True
    while must_generate_more_samples:
        # Do this loop in parallel using ProcessPoolExecutor.
        futures = list(map(
            lambda _: executor.submit(generate_train_action_examples,
                                      puzzle_train_examples, cell_value_size, discount, padding_char),
            range(cpu_count)))
        list_of_samples = list(map(lambda future: future.result(), futures))
        for samples in list_of_samples:
            writer.append(samples)
            generated_samples += len(samples)
            if generated_samples >= total_train_samples:
                must_generate_more_samples = False
                break


# TODO this function should receive just one puzzle example


def generate_train_action_examples(puzzle_examples, cell_value_size, discount: float, padding_char: str):
    """
    Generate (state, action, action_value) experience samples for all puzzle examples.
    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value)
    until the end of the game is reached.
    This is essentially a wrong incorrect half-baked MCTS (Monte Carlo tree search) in the sense that it's a
    tree search. And there is no Monte Carlo or Markov Chain involved here since we are lazy at best.
    """
    train_examples = []
    for puzzle_example in puzzle_examples:
        action_examples = simulate_random_playout(
            puzzle_example, cell_value_size, discount, padding_char)
        train_examples += action_examples
    return train_examples


def actions_act_on_same_cell(action_1: QLearningAction, action_2: QLearningAction) -> bool:
    return (action_1.row(), action_1.col()) == (action_2.row(), action_2.col())


def simulate_random_playout(puzzle_example, cell_value_size, discount: float,
                            padding_char: str):
    """
    Generate (state, action, action_value) samples from a simulated playout of the puzzle by a player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.
    For the probability of selecting a correct color, the reasoning is that a cell can take 1 color out of 10 colors.
    So, we use a probability of 1 / 10 = 0.1
    With 10000018 samples, we will get 204082 playouts.
    For any cell, 10% of the 204082 playouts, or 20408 playouts, will have a correct value.

    See https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action
    See https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    """

    (raw_example_input, raw_example_output) = puzzle_example

    input_width = len(raw_example_input[0])
    input_height = len(raw_example_input)
    output_width = len(raw_example_output[0])
    output_height = len(raw_example_output)

    if (input_width, input_height) != (output_width, output_height):
        raise Exception(
            f"input and output have different sizes: {(input_width, input_height)} and {(output_width, output_height)}")

    translation_x = random.randrange(-input_width + 1, input_width)
    translation_y = random.randrange(-input_height + 1, input_height)

    example_input = translate_board(
        raw_example_input, translation_x, translation_y)
    example_output = translate_board(
        raw_example_output, translation_x, translation_y)

    samples = []
    example_input = get_puzzle_starting_state(
        example_input, "input_state")
    current_state = get_puzzle_starting_state(
        example_output, "current_state")

    # List the cells of the output grid in a random order.
    cells = []
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            cells.append([row, col])
    np.random.shuffle(cells)

    for cell in cells:
        row = cell[0]
        col = cell[1]
        # Use random cell pixel value.
        new_value = random.randrange(0, cell_value_size)
        candidate_action = QLearningAction(row, col, new_value)

        # We can select the legal action as soon as now.
        # An action assigns a correct color or an incorrect color to a cell.
        # The only thing that matters is that we shuffled the legal actions before selecting the action.

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

        sample = (input_text, action_value)
        samples.append(sample)

        current_state = next_state

    return samples


def get_puzzle_starting_state(state, mode: str) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = 0
            if mode == "input_state":
                value = state[row][col]
            current_state[row][col] = Cell(value)
    return current_state


def reward(expected_cell_value, cell_value) -> int:
    if expected_cell_value == cell_value:
        return 1.0
    else:
        return -1.0


def get_q_star_action_value(state, action: QLearningAction, example_output, discount) -> int:
    """
    - discount is gamma
    - Q*(s, a) = gamma^0 * r_{t+1} + gamma^1* r_{t+1} + gamma^2 * r_{t+2} + ...

    See https://www.science.org/doi/10.1126/science.153.3731.34
    See https://en.wikipedia.org/wiki/Bellman_equation
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
    input_state_text, current_state_text = get_state_texts(
        input_state, current_state, padding_char)

    action_text = ""
    action_text += "act" + "\n"
    action_text += action_to_text(current_state, action)

    return SampleInputTokens(
        text_to_tokens(input_state_text),
        text_to_tokens(current_state_text),
        text_to_tokens(action_text)
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def get_state_texts(input_state, current_state, padding_char: str):
    input_state_text = ""
    input_state_text += "ini" + "\n"
    input_state_text += input_state_to_text(input_state)

    current_state_text = ""
    current_state_text += "cur" + "\n"
    current_state_text += current_state_to_text(current_state)

    return input_state_text, current_state_text


def input_state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = state[row][col].value()
            output += str(value)
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


def translate_board(board, translation_x: int, translation_y: int):
    width = len(board[0])
    height = len(board)
    new_board = copy.deepcopy(board)
    for x in range(width):
        for y in range(height):
            new_board[y][x] = 0
    for src_x in range(width):
        dst_x = src_x + translation_x
        if dst_x < 0 or dst_x >= width:
            continue
        for src_y in range(height):
            dst_y = src_y + translation_y
            if dst_y < 0 or dst_y >= height:
                continue
            new_board[dst_y][dst_x] = board[src_y][src_x]
    return new_board
