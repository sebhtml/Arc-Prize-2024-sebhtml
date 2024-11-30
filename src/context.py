from file_storage import FileStorageWriter, SampleInputTokens
import random
import copy
import numpy as np
from typing import List, Tuple
from vision import Cell, do_visual_fixation
from vision import VACANT_CELL_VALUE,  MASKED_CELL_VALUE,  OUTSIDE_CELL_VALUE
from vision import VACANT_CELL_CHAR, MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR
from q_learning import QLearningAction, ReplayBuffer, Experience, GameState
from q_learning import reward, sum_of_future_rewards


def actions_act_on_same_cell(action_1: QLearningAction, action_2: QLearningAction) -> bool:
    return (action_1.row(), action_1.col()) == (action_2.row(), action_2.col())


def get_puzzle_starting_state(state, mode: str) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = VACANT_CELL_VALUE
            if mode == "input_state":
                value = state[row][col]
            current_state[row][col] = Cell(value)
    return current_state


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
