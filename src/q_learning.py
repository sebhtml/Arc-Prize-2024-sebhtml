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


def sum_of_future_rewards(immediate_reward: float, discount: float,
                          attented_current_state: List[List[Cell]],
                          attented_candidate_action: QLearningAction) -> float:
    """
    This software used reinforcement learning.
    It uses Q-learning.

    See https://en.wikipedia.org/wiki/Q-learning
    See https://en.wikipedia.org/wiki/Bellman_equation

    See https://www.science.org/doi/10.1126/science.153.3731.34
    """

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


def reward(expected_state: List[List[int]], candidate_action: QLearningAction) -> float:
    row = candidate_action.row()
    col = candidate_action.col()
    action_cell_value = candidate_action.cell_value()
    expected_cell_value = expected_state[row][col]
    if expected_cell_value == action_cell_value:
        return cell_match_reward
    else:
        return cell_mismatch_reward
