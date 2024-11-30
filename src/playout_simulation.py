from file_storage import FileStorageWriter, SampleInputTokens
import random
import copy
import numpy as np
from typing import List, Tuple
import concurrent.futures
import concurrent
from vision import Cell, do_visual_fixation
from vision import VACANT_CELL_VALUE, VACANT_CELL_CHAR, MASKED_CELL_VALUE, MASKED_CELL_CHAR, OUTSIDE_CELL_VALUE, OUTSIDE_CELL_CHAR
from q_learning import QLearningAction, ReplayBuffer, Experience, GameState
from q_learning import reward, sum_of_future_rewards


def generate_examples(train_dataset_path: str, total_train_examples: int, puzzle_train_examples, cell_value_size: int,
                      discount: float, padding_char: str, cpu_count: int):
    writer = FileStorageWriter(train_dataset_path)
    generated_examples = 0
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count)
    must_generate_more_examples = True
    while must_generate_more_examples:
        # Do this loop in parallel using ProcessPoolExecutor.
        futures = list(map(
            lambda _: executor.submit(generate_train_action_examples,
                                      puzzle_train_examples, cell_value_size, discount, padding_char),
            range(cpu_count)))
        list_of_examples = list(map(lambda future: future.result(), futures))
        for examples in list_of_examples:
            writer.append(examples)
            generated_examples += len(examples)
            if generated_examples >= total_train_examples:
                must_generate_more_examples = False
                break


# TODO this function should receive just one puzzle example


def generate_train_action_examples(puzzle_examples, cell_value_size, discount: float, padding_char: str):
    """
    Generate (state, action, action_value) experience examples for all puzzle examples.
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
