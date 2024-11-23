from file_storage import FileStorageWriter, SampleInputTokens
import random
import copy
import numpy as np
from typing import List, Tuple
import concurrent.futures
import concurrent
from vision import Cell, do_circular_shift
from vision import VACANT_CELL_VALUE, VACANT_CELL_CHAR, MASKED_CELL_VALUE, MASKED_CELL_CHAR, OUTSIDE_CELL_VALUE, OUTSIDE_CELL_CHAR


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


class GameState:
    def __init__(self, example_input: List[List[Cell]], current_state: List[List[Cell]]):
        self.__example_input = example_input
        self.__current_state = current_state

    def example_input(self) -> List[List[Cell]]:
        return self.__example_input

    def current_state(self) -> List[List[Cell]]:
        return self.__current_state


class Experience:
    def __init__(self, state: GameState, action: QLearningAction, reward: float, next_state: GameState):
        self.__state = state
        self.__action = action
        self.__reward = reward
        self.__next_state = next_state

    def state(self) -> GameState:
        return self.__state

    def reward(self) -> float:
        return self.__reward

    def action(self) -> QLearningAction:
        return self.__action


class ReplayBuffer:
    """
    See https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action
    """

    def __init__(self):
        self.__experiences: List[Experience] = []

    def add_experience(self, experience: Experience):
        self.__experiences.append(experience)

    def experiences(self) -> List[Experience]:
        return self.__experiences


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
        replay_buffer = simulate_random_game(
            puzzle_example, cell_value_size)
        action_examples = extract_action_examples(
            replay_buffer, discount, padding_char)
        train_examples += action_examples
    return train_examples


def sum_of_future_rewards(immediate_reward: float, discount: float,
                          attented_current_state: List[List[Cell]],
                          attented_candidate_action: QLearningAction) -> float:
    expected_rewards = 0.0
    t = 0

    discounted_reward = discount**t * immediate_reward
    expected_rewards += discounted_reward
    t += 1

    # Count the number of remaining actions in the glimpe of the visual fixation.
    cells_that_can_change = 0
    for row in range(len(attented_current_state)):
        for col in range(len(attented_current_state[row])):
            # Skip cell because it was already counted as the immediate reward.
            if row == attented_candidate_action.row() and col == attented_candidate_action.col():
                continue
            # A cell can only be changed once.
            # TODO don't count the cells outside of the puzzle board.
            if attented_current_state[row][col].value() != VACANT_CELL_VALUE:
                continue
            cells_that_can_change += 1

    for _ in range(cells_that_can_change):
        # assume perfect play
        future_reward = cell_match_reward
        discounted_reward = discount**t * future_reward
        expected_rewards += discounted_reward
        t += 1

    return expected_rewards


def extract_action_examples(replay_buffer: ReplayBuffer, discount: float, padding_char: str) -> List[Tuple[SampleInputTokens, float]]:
    """
    This software used reinforcement learning.
    It uses Q-learning.

    See https://en.wikipedia.org/wiki/Q-learning
    See https://en.wikipedia.org/wiki/Bellman_equation

    See https://www.science.org/doi/10.1126/science.153.3731.34
    """

    examples = []

    experiences = replay_buffer.experiences()
    for experience in experiences:
        immediate_reward = experience.reward()
        example_input = experience.state().example_input()
        current_state = experience.state().current_state()
        candidate_action = experience.action()

        (attented_example_input, attented_current_state, attented_candidate_action,
         translation_x, translation_y) = do_visual_fixation(
            example_input, current_state, candidate_action)

        # attented_current_state = mask_cells(
        # current_state, attented_current_state)

        input_tokens = tokenize_sample_input(
            attented_example_input, attented_current_state, attented_candidate_action, padding_char)

        expected_rewards = sum_of_future_rewards(
            immediate_reward, discount, attented_current_state, attented_candidate_action)
        action_value = expected_rewards
        example = (input_tokens, action_value)

        examples.append(example)

    return examples


def actions_act_on_same_cell(action_1: QLearningAction, action_2: QLearningAction) -> bool:
    return (action_1.row(), action_1.col()) == (action_2.row(), action_2.col())


def simulate_random_game(puzzle_example, cell_value_size) -> ReplayBuffer:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by a random player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.
    For the probability of selecting a correct color, the reasoning is that a cell can take 1 color out of 10 colors.
    So, we use a probability of 1 / 10 = 0.1

    For any cell, 10% of the random games will have a correct value for that cell.
    """

    replay_buffer = ReplayBuffer()

    (raw_example_input, raw_example_output) = puzzle_example

    input_width = len(raw_example_input[0])
    input_height = len(raw_example_input)
    output_width = len(raw_example_output[0])
    output_height = len(raw_example_output)

    if (input_width, input_height) != (output_width, output_height):
        raise Exception(
            f"input and output have different sizes: {(input_width, input_height)} and {(output_width, output_height)}")

    example_input = raw_example_input
    example_output = raw_example_output

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

        # An action assigns a correct color or an incorrect color to a cell.
        # The only thing that matters is that we shuffled the legal actions before selecting the action.

        next_state = copy.deepcopy(current_state)
        next_state[row][col].set_value(new_value)

        immediate_reward = reward(example_output, candidate_action)

        experience = Experience(
            GameState(example_input, current_state),
            candidate_action,
            immediate_reward,
            GameState(example_input, next_state),
        )
        replay_buffer.add_experience(experience)
        current_state = next_state

    return replay_buffer


def get_puzzle_starting_state(state, mode: str) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = VACANT_CELL_VALUE
            if mode == "input_state":
                value = state[row][col]
            current_state[row][col] = Cell(value)
    return current_state


cell_match_reward = 1.0
cell_mismatch_reward = -1.0


def reward(expected_state: List[List[int]], candidate_action: QLearningAction) -> float:
    row = candidate_action.row()
    col = candidate_action.col()
    action_cell_value = candidate_action.cell_value()
    expected_cell_value = expected_state[row][col]
    if expected_cell_value == action_cell_value:
        return cell_match_reward
    else:
        return cell_mismatch_reward


def generate_cell_actions(current_state, cell_value_size) -> list[QLearningAction]:
    candidate_actions = []
    assert current_state != None
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].value() != VACANT_CELL_VALUE:
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


def tokens_to_text(sample_input_tokens: SampleInputTokens) -> str:
    tokens: List[int] = sample_input_tokens._input_state + \
        sample_input_tokens._current_state + sample_input_tokens._action
    return "".join(map(chr, tokens))


def get_state_texts(input_state, current_state, padding_char: str):
    input_state_text = ""
    input_state_text += "ini" + "\n"
    input_state_text += state_to_text(input_state)

    current_state_text = ""
    current_state_text += "cur" + "\n"
    current_state_text += state_to_text(current_state)

    return input_state_text, current_state_text


def state_to_text(state) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if state[row][col].value() == VACANT_CELL_VALUE:
                value = VACANT_CELL_CHAR
            elif state[row][col].value() == MASKED_CELL_VALUE:
                value = MASKED_CELL_CHAR
            elif state[row][col].value() == OUTSIDE_CELL_VALUE:
                value = OUTSIDE_CELL_CHAR
            else:
                value = str(state[row][col].value())
            output += value
        output += "\n"
    return output


def action_to_text(state, action: QLearningAction) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if row == action.row() and col == action.col():
                value = str(action.cell_value())
            else:
                value = VACANT_CELL_CHAR
            output += value
        output += "\n"
    return output


def translate_board(board, translation_x: int, translation_y: int, default_cell=0):
    """
    default_cell is 0 or Cell(0)
    """
    width = len(board[0])
    height = len(board)
    new_board = copy.deepcopy(board)
    for x in range(width):
        for y in range(height):
            new_board[y][x] = default_cell
    for src_x in range(width):
        dst_x = src_x + translation_x
        if dst_x < 0 or dst_x >= width:
            continue
        for src_y in range(height):
            dst_y = src_y + translation_y
            if dst_y < 0 or dst_y >= height:
                continue
            new_board[dst_y][dst_x] = copy.deepcopy(board[src_y][src_x])
    return new_board


def do_visual_fixation(example_input, current_state, candidate_action: QLearningAction):
    """
    Attend to the cell that is changed by the action.
    To do so, make the vision system put that cell in the center
    of the field of view.

    See
    Learning to combine foveal glimpses with a third-order Boltzmann machine
    https://www.cs.toronto.edu/~hinton/absps/nips_eyebm.pdf

    See https://en.wikipedia.org/wiki/Fixation_(visual)
    """

    input_height = len(example_input)
    input_width = len(example_input[0])

    row = candidate_action.row()
    col = candidate_action.col()
    new_value = candidate_action.cell_value()

    center_x = input_width // 2
    center_y = input_height // 2

    translation_x = center_x - col
    translation_y = center_y - row

    attented_example_input = do_circular_shift(
        example_input, translation_x, translation_y)
    attented_current_state = do_circular_shift(
        current_state, translation_x, translation_y)
    attented_candidate_action = QLearningAction(
        center_y, center_x, new_value)

    return [
        attented_example_input,
        attented_current_state,
        attented_candidate_action,
        translation_x,
        translation_y,
    ]
