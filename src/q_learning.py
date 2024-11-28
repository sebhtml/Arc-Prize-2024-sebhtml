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
