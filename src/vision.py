from typing import List, Tuple
import torch
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            if board[y][x].cell_value() == VACANT_CELL_VALUE:
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

    if input_height % 2 == 0 or input_width % 2 == 0:
        raise Exception("unable to do visual fixation")

    row = cell_address.row()
    col = cell_address.col()

    center_x = input_width // 2
    center_y = input_height // 2

    translation_x = center_x - col
    translation_y = center_y - row

    attented_example_input = translate_board(
        example_input, translation_x, translation_y, Cell(0))

    return attented_example_input


def crop_field_of_view(view: List[List[Cell]], visual_fixation_width: int, visual_fixation_height: int,) -> List[List[Cell]]:
    current_height = len(view)
    current_width = len(view[0])
    if current_height % 2 == 0 or current_width % 2 == 0:
        raise Exception("unable to crop")

    if visual_fixation_width > current_width or visual_fixation_height > current_height:
        raise Exception("unable to crop")

    if visual_fixation_height == current_height and visual_fixation_width == current_width:
        return view

    width_to_crop = current_width - visual_fixation_width
    half_width_to_crop = width_to_crop // 2
    height_to_crop = current_height - visual_fixation_height
    half_height_to_crop = height_to_crop // 2

    # crop height
    start = half_height_to_crop
    end = current_height - half_height_to_crop
    view = view[start:end]

    output = []
    for row in view:
        # crop width
        start = half_width_to_crop
        end = current_width - half_width_to_crop
        row = row[start:end]
        output.append(row)

    return output


def add_cell_saliency(state: List[List[Cell]], saliency: List[List[int]]):
    """
    Set saliency in state cells.
    """
    height = len(state)
    width = len(state[0])

    for x in range(width):
        for y in range(height):
            state[y][x].set_saliency(saliency[y][x])


def get_saliency(state: List[List[int]], visual_fixation_width: int, visual_fixation_height: int, cell_value_size: int,) -> List[List[float]]:
    """
    Get saliency of cells.
    We use entropy as described by the litterature.

    See
    Saliency, Scale and Image Description
    https://homes.cs.washington.edu/~shapiro/EE596/notes/kadir.pdf
    """

    height = len(state)
    width = len(state[0])

    # Compute saliency.
    saliency = copy.deepcopy(state)

    q_values = []

    for x in range(width):
        for y in range(height):
            value = state[y][x]
            q_values.append(value)

    q_values = torch.tensor(q_values)
    q_counts = torch.bincount(q_values, minlength=cell_value_size)
    q = q_counts.float() / q_counts.sum()
    q_safe = torch.clamp(q, min=1e-10)

    for x in range(width):
        x_start = -(visual_fixation_width//2)
        x_stop = (visual_fixation_width//2) + 1
        # print(f"{x_start} {x_stop}")
        for y in range(height):
            # print(f"point {x},{y}")

            p_values = []

            y_start = -(visual_fixation_height//2)
            y_stop = (visual_fixation_height//2) + 1
            # print(f"{y_start} {y_stop}")
            for x_offset in range(x_start, x_stop):
                x_other = x + x_offset
                if x_other < 0 or x_other >= width:
                    continue
                for y_offset in range(y_start, y_stop):
                    y_other = y + y_offset
                    if y_other < 0 or y_other >= height:
                        continue
                    # print(f"point {x_other},{y_other}")

                    value = state[y_other][x_other]
                    p_values.append(value)

            # print(f"p_values {len(p_values)}")

            p_values = torch.tensor(p_values)
            p_counts = torch.bincount(p_values, minlength=cell_value_size)
            p = p_counts.float() / p_counts.sum()
            p_safe = torch.clamp(p, min=1e-10)

            # print(f"p {p}")
            # print(f"q {q}")

            kl_div = (p * torch.log(p_safe / q_safe)).sum()
            entropy = -(p * torch.log(p_safe)).sum()
            cross_entropy = -(p * torch.log(q_safe)).sum()

            saliency[y][x] = entropy.item()

    return saliency


"""
https://en.wikipedia.org/wiki/Local_binary_patterns
"""


def get_local_binary_pattern(state: List[List[int]]) -> List[List[int]]:
    height = len(state)
    width = len(state[0])

    # Compute saliency.
    lbp = copy.deepcopy(state)
    for x in range(width):
        x_start = -1
        x_stop = 1 + 1
        # print(f"{x_start} {x_stop}")
        for y in range(height):
            center_value = state[y][x]
            # print(f"point {x},{y}")

            p_values = []

            y_start = -1
            y_stop = 1 + 1
            # print(f"{y_start} {y_stop}")

            pattern = []
            for x_offset in range(x_start, x_stop):
                x_other = x + x_offset
                for y_offset in range(y_start, y_stop):
                    if x_offset == 0 and y_offset == 0:
                        # Skip center.
                        continue
                    y_other = y + y_offset

                    if x_other < 0 or x_other >= width or y_other < 0 or y_other >= height:
                        pattern.append(0)
                        continue
                    # print(f"point {x_other},{y_other}")

                    neighbor_value = state[y_other][x_other]
                    value = 1 if neighbor_value > center_value else 0
                    pattern.append(value)

            # print(f"pattern {pattern}")
            integer = 0
            for i, bit in enumerate(pattern):
                # Calculate the value for each bit position
                integer += bit * 2**(7 - i)
            lbp[y][x] = integer

    return lbp


def center_surround_saliency(data: List[List[int]],) -> List[List[float]]:
    """
    Calculates a simple center-surround saliency map for a given image.

    Args:
      image: A 2D NumPy array representing the image.

    Returns:
      A 2D NumPy array representing the saliency map.

    See
    Receptive fields, binocular interaction and functional architecture in the cat's visual cortex
    The Journal of physiology, 1962•ncbi.nlm.nih.gov
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/pdf/jphysiol01247-0121.pdf?ref=hackernoon.com

    See
    A model of saliency-based visual attention for rapid scene analysis.
    IEEE Transactions on pattern analysis and machine intelligence, 1998•ieeexplore.ieee.org
    https://www.cse.psu.edu/~rtc12/CSE597E/papers/Itti_etal98pami.pdf
    """

    f = ReceptiveField()
    M = len(data)
    N = len(data[0])
    data_array = np.array(data)
    data_array = data_array.reshape(M, N)
    img = torch.from_numpy(data_array).unsqueeze(0).unsqueeze(0).float()

    output = f(img)

    return output.squeeze(0).squeeze(0).tolist()


class ReceptiveField(nn.Module):
    """
    See
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    
    See
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    """
    def __init__(self,):
        super(ReceptiveField, self).__init__()
        gaussian_kernel = [
            [1, 4, 7, 4, 1,],
            [4, 16, 26, 16, 4,],
            [7, 26, 41, 26, 7,],
            [4, 16, 26, 16, 4,],
            [1, 4, 7, 4, 1,],
        ]
        self.gaussian_kernel = torch.tensor(gaussian_kernel).float()
        self.gaussian_kernel = self.gaussian_kernel / self.gaussian_kernel.sum()
        self.gaussian_kernel = self.gaussian_kernel.unsqueeze(0).unsqueeze(0)

        laplacian_kernel = [
            [0, -1, 0,],
            [-1, 4, -1,],
            [0, -1, 0,],
        ]
        self.laplacian_kernel = torch.tensor(laplacian_kernel).float()
        self.laplacian_kernel = self.laplacian_kernel.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        #print("x")
        #print(x)
        
        #smoothed = nn.functional.conv2d(x, self.gaussian_kernel,
        #                              stride=1, padding="same")
        
        #print("smoothed")
        #print(smoothed)
        
        output = nn.functional.conv2d(x, self.laplacian_kernel,
                                      stride=1, padding="same")
        #print("output")
        #print(output)
        
        return output
