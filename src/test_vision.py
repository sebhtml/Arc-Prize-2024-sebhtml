from vision import rotate_90_clockwise, translate_board, flip_board


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
