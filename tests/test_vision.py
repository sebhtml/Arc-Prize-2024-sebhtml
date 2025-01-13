from vision import rotate_90_clockwise, translate_board, flip_board, crop_field_of_view
from vision import center_surround_receptive_field, do_visual_fixation
from vision import count_zero_crossings_2d, select_visual_fixations

import torch
import numpy as np
from torch import nn
import sys

from vision import CellAddress
from episode_renderer import print_state_with_colors
from main import make_celled_state
from configuration import Configuration


def test_rotate_90_clockwise():
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0, 0, 0],
        [0, 8, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 8, 0],
        [0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    new_board = rotate_90_clockwise(board)
    expected = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 8, 0],
        [0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 0, 0],
        [0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    assert new_board == expected


def test_translate_board_with_minus_one_x_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, -1, 0)
    expected = [[0, 0, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0], [8, 8, 0, 0, 0, 0, 0], [
        0, 0, 0, 8, 8, 0, 0], [0, 0, 0, 0, 8, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_translate_board_with_plus_one_x_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, +1, 0)
    expected = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0], [0, 0, 8, 8, 0, 0, 0], [
        0, 0, 0, 0, 0, 8, 8], [0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_translate_board_with_minus_one_y_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, 0, -1)
    expected = [[0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0], [
        0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_translate_board_with_plus_one_y_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, 0, 1)
    expected = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [
        0, 8, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_translate_board_with_minus_one_x_y_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, -1, -1)
    expected = [[8, 0, 0, 0, 0, 0, 0], [8, 8, 0, 0, 0, 0, 0], [0, 0, 0, 8, 8, 0, 0], [
        0, 0, 0, 0, 8, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_translate_board_with_plus_one_x_y_translation():
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 8, 0, 0, 0, 0, 0], [0, 8, 8, 0, 0, 0, 0], [
        0, 0, 0, 0, 8, 8, 0], [0, 0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    new_board = translate_board(board, +1, +1)
    expected = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0, 0], [
        0, 0, 8, 8, 0, 0, 0], [0, 0, 0, 0, 0, 8, 8], [0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0]]
    assert new_board == expected


def test_horizontal_flip():
    grid = [[1, 2, 3, 4], [5, 6, 7, 8]]

    # Horizontal flip
    flipped_horizontal = flip_board(grid, 'horizontal')
    assert flipped_horizontal == [[4, 3, 2, 1], [8, 7, 6, 5]]


def test_vertical_flip():
    grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
    # Vertical flip
    flipped_vertical = flip_board(grid, 'vertical')
    assert flipped_vertical == [[5, 6, 7, 8], [1, 2, 3, 4]]


def test_crop_field_of_view():
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0, 0, 0],
        [0, 8, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 8, 0],
        [0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    new_board = crop_field_of_view(board, 5, 5)
    expected = [
        [8, 0, 0, 0, 0],
        [8, 8, 0, 0, 0],
        [0, 0, 0, 8, 8],
        [0, 0, 0, 0, 8],
        [0, 0, 0, 0, 0],
    ]
    assert new_board == expected


def test_center_surround_receptive_field():
    config = Configuration()
    device = torch.device(config.device)

    state = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 8, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8, 0,],
        [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    print("state")
    for row in state:
        row2 = list(map(lambda x: str(round(x, 2)).rjust(6), row))
        print(row2)

    state = torch.tensor(state).float().unsqueeze(0).unsqueeze(0).to(device)
    laplacian = center_surround_receptive_field(state)

    expected = [
                [+0,  +0,  +0,  +0,  +0,  +0,  +0,  -8,  -8,  +0,  -8,  -8,  +0,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  +0,  +0,  -8, +24, +16, -24, +16, +24,  -8,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  +0,  -8,  -8, -24,  +8, +16, +16, -16,  +0,  +0],  # nopep8
                [+0,  +0,  +0,  +0, -16, +16, +16,  +8, +16, -16, -16,  -8,  +0,  +0],  # nopep8
                [+0,  +0,  +0,  -8, +24, +16, -24, +16, -24, -16, +16, +24,  -8,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  -8,  -8,  -8, +16,  +8, +16,  +8, -16,  +0,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  +0,  +0,  -8, -24, +16, -24, +16, -16,  -8,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  +0,  -8, +16, +16, +16, -16, +16, +16, +16,  -8],  # nopep8
                [+0,  +0,  -1,  +0,  +0,  -8, +24, -16,  -8,  +0,  -8, -16, +24,  -8],  # nopep8
                [+0,  -2,  +3,  -2,  +0,  +0,  -8,  +0,  +0,  +0,  +0,  +0,  -8,  +0],  # nopep8
                [-1,  +3,  +0,  +3,  -1,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0],  # nopep8
                [+0,  -2,  +3,  -2,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0],  # nopep8
                [+0,  +0,  -1,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0],  # nopep8
                [+0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0,  +0],  # nopep8
    ]
    expected = torch.tensor(expected).float().unsqueeze(0).unsqueeze(0)

    laplacian2 = laplacian.squeeze(0).squeeze(0).tolist()
    print("laplacian")
    for row in laplacian2:
        row2 = list(map(lambda x: str(round(x, 2)).rjust(6), row))
        print(row2)

    print(laplacian.tolist())
    print(expected.tolist())

    assert laplacian.tolist() == expected.tolist()


def test_count_zero_crossings_2d():
    config = Configuration()
    device = torch.device(config.device)

    # Example usage
    tensor = torch.tensor(
        [[[
            [-0.5558,  0.6872, -2.5284, -0.1288,  1.6507],
            [0.3000,  1.1827, -0.9129,  0.4243,  0.6307],
            [-1.1996, -0.7706, -3.8851, -0.1429, -2.0533],
            [-0.7130,  0.6620,  0.1569,  0.6320, -0.2912],
            [0.1315, -0.7237, -1.4901, -0.7380,  0.7858]
        ]]]
    ).to(device)

    expected = torch.tensor(
        [[[
            [2., 2., 1., 2., 1.],
            [2., 2., 2., 3., 1.],
            [1., 2., 1., 2., 1.],
            [2., 3., 2., 3., 2.],
            [2., 2., 1., 2., 2.]
        ]]]
    )
    edges = count_zero_crossings_2d(tensor)

    assert edges.tolist() == expected.tolist()


def test_select_visual_fixations():
    config = Configuration()
    device = torch.device(config.device)

    example_input = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 8, 8, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 8, 8, 0, 8, 0, 0, 8, 8, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 8, 8, 8, 0,],
        [0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    num_visual_fixations = 4
    visual_fixation_width = 5
    visual_fixation_height = 5
    addresses = select_visual_fixations(
        device,
        example_input, num_visual_fixations, visual_fixation_height, visual_fixation_width,)

    celled_example_input = make_celled_state(example_input)
    for cell_address in addresses:
        attented_example_input = do_visual_fixation(
            celled_example_input, cell_address, visual_fixation_height, visual_fixation_width,)

        print_state_with_colors(attented_example_input, sys.stdout,)

    simple_addresses = list(map(lambda a: [a.row(), a.col(),], addresses))

    assert simple_addresses == [[10, 2], [6, 10], [4, 5], [1, 10]]
    # assert True == False
