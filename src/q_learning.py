from typing import List


VACANT_CELL_VALUE = -1
MASKED_CELL_VALUE = -2
OUTSIDE_CELL_VALUE = -3

cell_match_reward = 1.0
cell_mismatch_reward = -1.0


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

    def value(self) -> int:
        return self.__value

    def set_value(self, value):
        self.__value = value


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

    def next_state(self) -> GameState:
        return self.__next_state


def reward(expected_state: List[List[int]], candidate_action: QLearningAction) -> float:
    row = candidate_action.row()
    col = candidate_action.col()
    action_cell_value = candidate_action.cell_value()
    expected_cell_value = expected_state[row][col]
    if expected_cell_value == action_cell_value:
        return cell_match_reward
    else:
        return cell_mismatch_reward
