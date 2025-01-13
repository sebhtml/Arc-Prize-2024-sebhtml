import torch
from typing import List
import math

from vision import Cell, CellAddress, do_visual_fixation
from vision import VACANT_CELL_VALUE, OUTSIDE_CELL_VALUE
from vision import VACANT_CELL_CHAR, OUTSIDE_CELL_CHAR

# There are 10 possible colors.
# 9 colors are incorrect and 1 color is correct.
# The control policy assigns a color to a cell.
# With a random policy, a given cell has 0.90 chance of being incorrect.
# And only 10% chance of being correct.
# Thus, since correct actions are rarer, they are rewarded more.
# This ensures that the agent will actually try to explore to reach those
# good rewards.
cell_match_reward = +10.0
cell_mismatch_reward = -1.0


def unbin_action_value(action_value_bin: int, minimum_action_value: float, maximum_action_value: float, num_classes: int) -> float:
    """
    Convert a bin between 0 and num_classes-1 to a float between minimum_action_value and maximum_action_value
    """
    minimum_action_value_bin = 0
    maximum_action_value_bin = num_classes - 1

    action_value = minimum_action_value + (
        ((action_value_bin - minimum_action_value_bin) / (maximum_action_value_bin - minimum_action_value_bin)) * (maximum_action_value - minimum_action_value))
    return max(min(action_value, maximum_action_value), minimum_action_value)


def bin_action_value(action_value: float, minimum_action_value: float, maximum_action_value: float, num_classes: int) -> int:
    """
    convert action_value to { 0, 1, ..., num_classes - 1 }
    Example:
    action_value: 3.0
    _minimum_action_value: -4.0
    _maximum_action_value: 7.0
    action_value_bin = (3.0 - -4.0) / (7.0 - -4.0)
    """
    minimum_action_value_bin = 0
    maximum_action_value_bin = num_classes - 1

    action_value_bin = minimum_action_value_bin + math.floor(
        ((action_value - minimum_action_value) / (maximum_action_value - minimum_action_value)) * (maximum_action_value_bin - minimum_action_value_bin))
    return max(min(action_value_bin, maximum_action_value_bin), minimum_action_value_bin)


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

    def __eq__(self, other) -> bool:
        return self.__row == other.__row and self.__col == other.__col and self.__cell_value == other.__cell_value


class ExampleInput:
    def __init__(self, cells: List[List[Cell]],):
        self.__cells = cells

    def cells(self) -> List[List[Cell]]:
        return self.__cells

    def visual_fixation_text(self, cell_address: CellAddress, visual_fixation_height: int, visual_fixation_width: int,) -> str:
        example_input = self.cells()
        visual_fixation = do_visual_fixation(
            example_input, cell_address, visual_fixation_height, visual_fixation_width,)
        return state_to_text(visual_fixation)


class GameState:
    def __init__(self, example_input: ExampleInput, current_state: List[List[Cell]]):
        self.__example_input = example_input
        self.__current_state = current_state

    def example_input(self) -> ExampleInput:
        return self.__example_input

    def current_state(self) -> List[List[Cell]]:
        return self.__current_state


class Experience:
    def __init__(self, state: GameState, action: QLearningAction, reward: float, next_state: GameState,
                 correct_action_index: int, log_probs: torch.Tensor,):
        self.__state = state
        self.__action = action
        self.__reward = reward
        self.__next_state = next_state
        self.__correct_action_index = correct_action_index
        self.__log_probs = log_probs

    def state(self) -> GameState:
        return self.__state

    def reward(self) -> float:
        return self.__reward

    def action(self) -> QLearningAction:
        return self.__action

    def next_state(self) -> GameState:
        return self.__next_state

    def correct_action_index(self) -> int:
        return self.__correct_action_index

    def log_probs(self) -> int:
        return self.__log_probs


def reward(expected_state: List[List[Cell]], candidate_action: QLearningAction) -> float:
    row = candidate_action.row()
    col = candidate_action.col()
    action_cell_value = candidate_action.cell_value()
    expected_cell_value = expected_state[row][col].cell_value()
    if expected_cell_value == action_cell_value:
        return cell_match_reward
    else:
        return cell_mismatch_reward


def trim_list(lst, k):
    """
    keep at most k elements from list lst
    """
    return lst[-k:] if len(lst) > k else lst


def state_to_text(state: List[List[Cell]]) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if state[row][col].cell_value() == VACANT_CELL_VALUE:
                value = VACANT_CELL_CHAR
            elif state[row][col].cell_value() == OUTSIDE_CELL_VALUE:
                value = OUTSIDE_CELL_CHAR
            else:
                value = str(state[row][col].cell_value())
            output += value
        output += "\n"
    return output
