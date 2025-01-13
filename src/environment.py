import random
from typing import List, Tuple, Union
from context import get_puzzle_starting_state
from q_learning import reward, QLearningAction, GameState, ExampleInput
from vision import VACANT_CELL_VALUE, Cell
import copy


class Environment:
    """
    A game environment for playing the game of ARC prize.

    See https://arcprize.org/play?task=3aa6fb7a
    See https://arcprize.org/
    """

    def __init__(self, cell_value_size: int):
        self.__example_input = None
        self.__example_output = None
        self.__current_state = None
        self.__cell_value_size = cell_value_size
        self.__available_actions = []
        # Metrics
        self.__total_rewards_per_episode = []
        self.__max_total_rewards_per_episode = []
        self.__current_episode_total_rewards = 0.0
        self.__recorded_episodes = []
        self.__current_episode = []

    def set_puzzle_example(self, example_input: ExampleInput, example_output: Union[List[List[Cell]], None]):
        """
        puzzle_output can be None. In that case, immediate reward will be None.
        """
        self.__example_output = example_output

        input_width = len(example_input.cells()[0])
        input_height = len(example_input.cells())

        if example_output != None:
            output_width = len(example_output[0])
            output_height = len(example_output)

            if (input_width, input_height) != (output_width, output_height):
                raise Exception(
                    f"input and output have different sizes: {(input_width, input_height)} and {(output_width, output_height)}")

        self.__example_input = example_input
        self.__current_state = get_puzzle_starting_state(example_input.cells())

        # Clear the current episode if the previous episode was truncated.
        self.__current_episode = []
        self.__current_episode.append(
            GameState(
                copy.deepcopy(self.__example_input),
                copy.deepcopy(self.__current_state),
            ))

        self.__available_actions = generate_cell_actions(
            self.__current_state, self.__cell_value_size)

        if self.is_in_terminal_state():
            self.__terminate_episode()

    def is_in_terminal_state(self) -> bool:
        return len(self.__available_actions) == 0

    def get_observations(self) -> Tuple[ExampleInput, List[List[Cell]]]:
        return (self.__example_input, self.__current_state)

    def take_action(self, action: QLearningAction) -> float:
        """
        Take an action and return a reward.
        """

        if not action in self.__available_actions:
            raise Exception(f"Illegal action provided: {action}.")

        row = action.row()
        col = action.col()
        new_value = action.cell_value()

        self.__current_state[row][col].set_cell_value(new_value)

        self.__current_episode.append(
            GameState(
                copy.deepcopy(self.__example_input),
                copy.deepcopy(self.__current_state),
            ))

        immediate_reward = None
        if self.__example_output != None:
            immediate_reward = reward(self.__example_output, action)
            self.__current_episode_total_rewards += immediate_reward

        self.__available_actions = generate_cell_actions(
            self.__current_state, self.__cell_value_size)

        if self.is_in_terminal_state():
            self.__terminate_episode()

        return immediate_reward

    def get_optimal_action_index(self, row: int, col: int) -> int:
        """
        Get the correct action.
        """

        if self.__example_output == None:
            return None

        expected_state = self.__example_output

        expected_cell_value = expected_state[row][col].cell_value()

        return expected_cell_value

    def list_actions(self) -> List[QLearningAction]:
        return self.__available_actions

    def __terminate_episode(self):
        self.__recorded_episodes.append(self.__current_episode)
        self.__current_episode = []
        self.__total_rewards_per_episode.append(
            self.__current_episode_total_rewards)
        self.__max_total_rewards_per_episode.append(
            self.get_max_total_rewards())
        self.__current_episode_total_rewards = 0.0

    def get_total_rewards_per_episode(self):
        return self.__total_rewards_per_episode

    def get_max_total_rewards_per_episode(self):
        return self.__max_total_rewards_per_episode

    def recorded_episodes(self) -> List[List[GameState]]:
        return self.__recorded_episodes

    def get_max_total_rewards(self) -> int:
        """
        Get the maximum number of rewards for the current puzzle example.
        """
        if self.__example_output != None:
            output_width = len(self.__example_output[0])
            output_height = len(self.__example_output)
            return output_height * output_width
        return None


def generate_cell_actions(
        current_state: List[List[Cell]],
        cell_value_size: int,
) -> List[QLearningAction]:

    candidate_actions = []
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].cell_value() == VACANT_CELL_VALUE:
                for new_value in range(cell_value_size):
                    action = QLearningAction(row, col, new_value)
                    candidate_actions.append(action)

    return candidate_actions
