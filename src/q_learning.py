from typing import List
import math

VACANT_CELL_VALUE = -1
MASKED_CELL_VALUE = -2
OUTSIDE_CELL_VALUE = -3

cell_match_reward = +1.0
cell_mismatch_reward = 0.0


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


class CellAddress:
    def __init__(self, row: int, col: int,):
        self.__row = row
        self.__col = col

    def row(self) -> int:
        return self.__row

    def col(self) -> int:
        return self.__col


class Cell:
    def __init__(self, value):
        self.__value = value

    def value(self) -> int:
        return self.__value

    def set_value(self, value):
        self.__value = value

    def __eq__(self, other) -> bool:
        return self.__value == other.__value


class GameState:
    def __init__(self, example_input: List[List[Cell]], current_state: List[List[Cell]]):
        self.__example_input = example_input
        self.__current_state = current_state

    def example_input(self) -> List[List[Cell]]:
        return self.__example_input

    def current_state(self) -> List[List[Cell]]:
        return self.__current_state


class Experience:
    def __init__(self, state: GameState, action: QLearningAction, reward: float, next_state: GameState,
                 correct_action_index: int):
        self.__state = state
        self.__action = action
        self.__reward = reward
        self.__next_state = next_state
        self.__correct_action_index = correct_action_index

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


def reward(expected_state: List[List[int]], candidate_action: QLearningAction) -> float:
    row = candidate_action.row()
    col = candidate_action.col()
    action_cell_value = candidate_action.cell_value()
    expected_cell_value = expected_state[row][col]
    if expected_cell_value == action_cell_value:
        return cell_match_reward
    else:
        return cell_mismatch_reward
