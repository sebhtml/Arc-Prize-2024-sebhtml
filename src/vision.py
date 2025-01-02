from typing import List, Tuple
import numpy as np
import copy
import random
from q_learning import QLearningAction, Cell, CellAddress
from q_learning import VACANT_CELL_VALUE, OUTSIDE_CELL_VALUE


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

    See https://pytorch.org/vision/main/generated/torchvision.transforms.RandomRotation.html

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


def flip_board(grid, direction):
    """Flips a 2D grid horizontally or vertically.

    See https://help.optitex.com/1382687/Content/Marker/Flip_Horizontal.htm
    See https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html
    See https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html

    Args:
    grid: A 2D list representing the grid.
    direction: A string, either 'horizontal' or 'vertical', indicating the direction of the flip.

    Returns:
    A new 2D list representing the flipped grid.
    """

    if direction == 'horizontal':
        return [row[::-1] for row in grid]
    elif direction == 'vertical':
        return grid[::-1]
    else:
        raise ValueError(
            "Invalid direction. Must be 'horizontal' or 'vertical'.")


def do_visual_fixation(example_input: List[List[Cell]], cell_address: CellAddress) -> List[List[Cell]]:
    """
    Attend to a cell address.

    To do so, make the vision system put that cell in the center
    of the field of view.

    See
    Learning to combine foveal glimpses with a third-order Boltzmann machine
    https://www.cs.toronto.edu/~hinton/absps/nips_eyebm.pdf

    See https://en.wikipedia.org/wiki/Fixation_(visual)

    See
    Predicting human gaze beyond pixels
    https://jov.arvojournals.org/article.aspx?articleid=2193943
    """

    input_height = len(example_input)
    input_width = len(example_input[0])

    row = cell_address.row()
    col = cell_address.col()

    center_x = input_width // 2
    center_y = input_height // 2

    translation_x = center_x - col
    translation_y = center_y - row

    attented_example_input = translate_board(
        example_input, translation_x, translation_y, Cell(0))

    return attented_example_input


def crop_field_of_view(view: List[List[Cell]], crop_width: int, crop_height: int,) -> List[List[Cell]]:
    # TODO
    return view
