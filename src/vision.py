from typing import List, Tuple
import torch
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

VACANT_CELL_VALUE = -1
OUTSIDE_CELL_VALUE = -3

VACANT_CELL_CHAR = '_'
OUTSIDE_CELL_CHAR = '.'


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
        self.__saliency = 0

    def cell_value(self) -> int:
        return self.__value

    def set_cell_value(self, value):
        self.__value = value

    def __eq__(self, other) -> bool:
        return self.__value == other.__value

    def set_saliency(self, saliency: float):
        self.__saliency = saliency

    def saliency(self) -> float:
        """
        https://en.wikipedia.org/wiki/Salience_(neuroscience)
        """
        return self.__saliency


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


def do_visual_fixation(example_input: List[List[Cell]], cell_address: CellAddress,
                       visual_fixation_height: int, visual_fixation_width: int,
                       ) -> List[List[Cell]]:
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

    example_input = pad_state(example_input)

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

    attented_example_input = crop_field_of_view(
        attented_example_input, visual_fixation_height, visual_fixation_width,)

    return attented_example_input


def crop_field_of_view(view: List[List[Cell]],
                       visual_fixation_height: int,
                       visual_fixation_width: int,
                       ) -> List[List[Cell]]:
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


def center_surround_receptive_field(img: torch.Tensor,) -> torch.Tensor:
    """
    Calculates a simple center-surround saliency map for a given image.

    Args:
      img: a tensor of shape [1,1,M,N]

    Returns:
      a tensor of shape [1,1,M,N]

    See
    Receptive fields, binocular interaction and functional architecture in the cat's visual cortex
    The Journal of physiology, 1962•ncbi.nlm.nih.gov
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/pdf/jphysiol01247-0121.pdf?ref=hackernoon.com

    See
    A model of saliency-based visual attention for rapid scene analysis.
    IEEE Transactions on pattern analysis and machine intelligence, 1998•ieeexplore.ieee.org
    https://www.cse.psu.edu/~rtc12/CSE597E/papers/Itti_etal98pami.pdf
    """

    f = ReceptiveField(img.device)
    output = f(img)
    return output


class ReceptiveField(nn.Module):
    """
    Use a Laplacian kernel to detect edges in a receptive field.

    See
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm

    See
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    """

    def __init__(self, device: torch.device,):
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
        self.gaussian_kernel = self.gaussian_kernel.unsqueeze(
            0).unsqueeze(0).to(device)

        laplacian_kernel = [
            [0, -1, 0,],
            [-1, 4, -1,],
            [0, -1, 0,],
        ]
        self.laplacian_kernel = torch.tensor(laplacian_kernel).float()
        self.laplacian_kernel = self.laplacian_kernel.unsqueeze(
            0).unsqueeze(0).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.conv2d(x, self.laplacian_kernel,
                                      stride=1, padding="same")
        return output


def count_zero_crossings_2d(tensor):
    """
    Counts zero crossings in both x and y axes of a 4D PyTorch tensor 
    (shape: [1, 1, M, N]), and returns tensors indicating locations of crossings.

    Args:
      tensor: Input tensor with shape [1, 1, M, N].

    Returns:
      A tuple containing:
        - sign_changes: A tensor with shape [M, N] representing locations of x-axis zero crossings (1 for crossing, 0 otherwise).
    """

    device = tensor.device

    # Squeeze redundant dimensions
    tensor = tensor.squeeze(0).squeeze(0)  # Now shape is [M, N]

    # Calculate sign of each element
    sign = torch.sign(tensor)

    # Detect sign changes (avoiding out-of-bounds access)
    sign_changes = torch.zeros_like(sign).to(device)  # Initialize with zeros
    sign_changes[:, 1:] += sign[:, :-1] != sign[:, 1:]
    sign_changes[:, :-1] += sign[:, 1:] != sign[:, :-1]
    sign_changes[1:, :] += sign[:-1, :] != sign[1:, :]
    sign_changes[:-1, :] += sign[1:, :] != sign[:-1, :]

    return sign_changes.unsqueeze(0).unsqueeze(0)


def select_visual_fixations(
    device: torch.device,
    state: List[List[int]],
    num_visual_fixations: int,
    visual_fixation_height: int, visual_fixation_width: int,
) -> List[CellAddress]:

    state = torch.tensor(state).float().unsqueeze(0).unsqueeze(0).to(device)
    laplacian = center_surround_receptive_field(state)
    edges = count_zero_crossings_2d(laplacian)

    cell_addresses = []

    while len(cell_addresses) < num_visual_fixations:
        # edges2 = edges.squeeze(0).squeeze(0).tolist()
        # print("edges")
        # for row in edges2:
        #    row2 = list(map(lambda x: str(round(x, 2)).rjust(6), row))
        #    print(row2)

        saliency_kernel = torch.ones(
            1, 1, visual_fixation_height, visual_fixation_width).to(device)
        saliency = nn.functional.conv2d(edges, saliency_kernel,
                                        stride=1, padding="same")

        # saliency2 = saliency.squeeze(0).squeeze(0).tolist()
        # print("saliency")
        # for row in saliency2:
        #    row2 = list(map(lambda x: str(round(x, 2)).rjust(6), row))
        #    print(row2)

        N = saliency.shape[-1]
        flattened = saliency.view(-1)
        max_index = flattened.argmax()
        y, x = max_index // N, max_index % N
        y, x = y.item(), x.item()

        # print(f"max at y,x={y,x}")
        addr = CellAddress(y, x)
        cell_addresses.append(addr)
        edges[:, :, y-(visual_fixation_height//2):y+(visual_fixation_height//2)+1, x -
              (visual_fixation_width//2):x+(visual_fixation_width//2)+1] = 0.0

    return cell_addresses


def pad_state(example_input: List[List[Cell]],) -> List[List[Cell]]:
    example_input = copy.deepcopy(example_input)
    input_height = len(example_input)
    input_width = len(example_input[0])

    # if height is even, add a dummy row.
    if input_height % 2 == 0:
        example_input.append([Cell(0)] * input_width)
        input_height = len(example_input)

    # if width is even, add a dummy column
    if input_width % 2 == 0:
        for row in example_input:
            row.append(Cell(0))
        input_width = len(example_input[0])

    return example_input
