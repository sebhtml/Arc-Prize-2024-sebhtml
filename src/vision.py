from typing import List, Tuple
import numpy as np
import copy
import random
from q_learning import QLearningAction, Cell

VACANT_CELL_VALUE = -1
MASKED_CELL_VALUE = -2
OUTSIDE_CELL_VALUE = -3

VACANT_CELL_CHAR = '_'
MASKED_CELL_CHAR = 'X'
OUTSIDE_CELL_CHAR = '.'


def get_total_cells(board: List[List[Cell]]) -> int:
    width = len(board[0])
    height = len(board)
    return width * height


def get_vacant_cells(board: List[List[Cell]]) -> List[Tuple[int, int]]:
    vacant_cells = []
    width = len(board[0])
    height = len(board)
    for x in range(width):
        for y in range(height):
            if board[y][x].value() == VACANT_CELL_VALUE:
                vacant_cells.append((x, y))
    return vacant_cells


def get_visible_cells_in_smallest_fixation(board: List[List[Cell]]) -> int:
    width = len(board[0])
    height = len(board)

    # 7 => 3
    center_x = width // 2
    center_y = height // 2

    # (7, 7) => 4*4 = 16
    visible_width = width - center_x
    visible_height = height - center_y
    return visible_width * visible_height


def mask_cells(current_state: List[List[Cell]], attented_current_state: List[List[Cell]]) -> List[List[Cell]]:
    """
    Vacant cells are used to estimate the future expected rewards.
    Fixations with more vacant cells will have a higher future expected sum of rewards.

    Visual fixations in the corners of the board have a lower amount of expected vacant cells,
    when compared to the visual fixation looking at the center cell.
    For example, for a 7x7 grid, the centered visual fixation has a maximum of 49 vacant cells,
    while a corner fixation only has 16 vacant cells.

    We need to mask some vacant cells in order for all the visual fixations of a given board state
    to have roughly the same number of non-masked vacant cells.
    """

    total_cells = get_total_cells(current_state)
    smallest_fixation_cells = get_visible_cells_in_smallest_fixation(
        current_state)
    vacant_cells = len(get_vacant_cells(current_state))
    maximum_unmasked_vacant_cells = (
        vacant_cells / total_cells) * smallest_fixation_cells
    maximum_unmasked_vacant_cells = int(maximum_unmasked_vacant_cells)

    # The center cell is always allowed to be vacant.
    if maximum_unmasked_vacant_cells <= 1:
        return attented_current_state

    width = len(attented_current_state[0])
    height = len(attented_current_state)

    # 7 => 3
    center_x = width // 2
    center_y = height // 2
    center = (center_x, center_y)

    if attented_current_state[center_y][center_x].value() != VACANT_CELL_VALUE:
        raise Exception("Center cell must be vacant")

    vacant_cells = get_vacant_cells(attented_current_state)

    # When the number of vacant cells in the visual fixation is lower than
    # the maximum number ofunmasked vacant cells, then it is fine.
    if len(vacant_cells) <= maximum_unmasked_vacant_cells:
        return attented_current_state

    vacant_cells = list(filter(lambda cell: cell != center, vacant_cells))
    maximum_unmasked_vacant_cells -= 1

    # Mask some vacant cells.
    np.random.shuffle(vacant_cells)

    # We need to mask some vacant cells such that the number
    # of unmasked vacant cells is below or equal to the maximum.
    n = len(vacant_cells) - maximum_unmasked_vacant_cells
    for _ in range(n):
        x, y = vacant_cells.pop()
        attented_current_state[y][x].set_value(MASKED_CELL_VALUE)

    return attented_current_state


def do_circular_shift(board, shift_x: int, shift_y: int):
    """
    https://en.wikipedia.org/wiki/Circular_shift
    """
    width = len(board[0])
    height = len(board)
    new_board = copy.deepcopy(board)
    for src_x in range(width):
        dst_x = (src_x + shift_x) % width
        for src_y in range(height):
            dst_y = (src_y + shift_y) % height
            new_board[dst_y][dst_x] = copy.deepcopy(board[src_y][src_x])
    return new_board


def rotate_90_clockwise(grid):
    """Rotates a 2D grid 90 degrees clockwise.

    Args:
    grid: A 2D list representing the grid.

    Returns:
    A new 2D list representing the rotated grid.
    """

    n = len(grid)
    rotated_grid = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            rotated_grid[j][n - i - 1] = grid[i][j]

    return rotated_grid


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

    attented_example_input = translate_board(
        example_input, translation_x, translation_y, Cell(OUTSIDE_CELL_VALUE))
    attented_current_state = translate_board(
        current_state, translation_x, translation_y, Cell(OUTSIDE_CELL_VALUE))
    attented_candidate_action = QLearningAction(
        center_y, center_x, new_value)

    attented_current_state = mask_cells(
        current_state, attented_current_state)

    rotations = random.randrange(0, 4)

    for _ in range(rotations):
        attented_example_input = rotate_90_clockwise(attented_example_input)
        attented_current_state = rotate_90_clockwise(attented_current_state)

    return [
        attented_example_input,
        attented_current_state,
        attented_candidate_action,
        translation_x,
        translation_y,
    ]
