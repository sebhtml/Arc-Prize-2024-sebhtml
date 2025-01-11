from vision import rotate_90_clockwise, translate_board, flip_board, crop_field_of_view
from vision import center_surround_receptive_field
from vision import count_zero_crossings_2d
import torch
import numpy as np


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


def test_add_cell_saliency():
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

    laplacian = center_surround_receptive_field(np.array(state),)

    print("laplacian")
    for row in laplacian:
        row2 = list(map(lambda x: str(round(x, 2)).rjust(6), row))
        print(row2)

    # edges = count_edges(laplacian)

    assert False == True


def test_count_zero_crossings_2d():
    # Example usage
    tensor = torch.tensor(
        [[[
            [-0.5558,  0.6872, -2.5284, -0.1288,  1.6507],
            [0.3000,  1.1827, -0.9129,  0.4243,  0.6307],
            [-1.1996, -0.7706, -3.8851, -0.1429, -2.0533],
            [-0.7130,  0.6620,  0.1569,  0.6320, -0.2912],
            [0.1315, -0.7237, -1.4901, -0.7380,  0.7858]
          ]]]
    )

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
