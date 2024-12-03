import random
from typing import List, Tuple, Union
from context import get_puzzle_starting_state
from q_learning import Cell, reward, QLearningAction
from vision import VACANT_CELL_VALUE


class Emulator:
    """
    A game emulator for playing the game of ARC prize.

    See https://arcprize.org/play?task=3aa6fb7a
    See https://arcprize.org/
    """

    def __init__(self, cell_value_size: int):
        self.__puzzle_input = None
        self.__puzzle_output = None
        self.__example_input = None
        self.__current_state = None
        self.__cell_value_size = cell_value_size

    def set_puzzle_example(self, puzzle_input: List[List[int]], puzzle_output: Union[List[List[int]], None]):
        """
        puzzle_output can be None. In that case, immediate reward will be None.
        """
        self.__puzzle_input = puzzle_input
        self.__puzzle_output = puzzle_output

        input_width = len(self.__puzzle_input[0])
        input_height = len(self.__puzzle_input)

        if self.__puzzle_output != None:
            output_width = len(self.__puzzle_output[0])
            output_height = len(self.__puzzle_output)

            if (input_width, input_height) != (output_width, output_height):
                raise Exception(
                    f"input and output have different sizes: {(input_width, input_height)} and {(output_width, output_height)}")

        self.__example_input = get_puzzle_starting_state(
            self.__puzzle_input, "input_state")
        self.__current_state = get_puzzle_starting_state(
            self.__puzzle_input, "current_state")

        self.__available_actions = generate_cell_actions(
            self.__current_state, self.__cell_value_size)

    def is_in_terminal_state(self) -> bool:
        return len(self.__available_actions) == 0

    def game_state(self) -> Tuple[List[List[Cell]], List[List[Cell]]]:
        return (self.__example_input, self.__current_state)

    def take_action(self, action: QLearningAction) -> float:
        """
        Take an action and return a reward.
        """

        if not action in self.__available_actions:
            raise Exception("Illegal action provided.")

        row = action.row()
        col = action.col()
        new_value = action.cell_value()

        self.__current_state[row][col].set_value(new_value)

        immediate_reward = None
        if self.__puzzle_output != None:
            immediate_reward = reward(self.__puzzle_output, action)

        self.__available_actions = generate_cell_actions(
            self.__current_state, self.__cell_value_size)

        return immediate_reward

    def list_actions(self) -> list[QLearningAction]:
        return self.__available_actions


def generate_cell_actions(current_state, cell_value_size) -> list[QLearningAction]:

    vacant_cells = []
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].value() == VACANT_CELL_VALUE:
                vacant_cells.append([row, col])

    if len(vacant_cells) == 0:
        return []

    # Select randomly a cell to be filled.
    i = random.randrange(0, len(vacant_cells))

    vacant_cell = vacant_cells[i]
    row = vacant_cell[0]
    col = vacant_cell[1]

    candidate_actions = []
    for new_value in range(cell_value_size):
        action = QLearningAction(row, col, new_value)
        candidate_actions.append(action)

    return candidate_actions
