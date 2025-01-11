from training import bin_action_value, unbin_action_value

#
# bin_action_value
#


def test_bin_action_value_middle():
    assert bin_action_value(33.0, -50.0, +50.0, 128) == 105


def test_bin_action_value_first():
    assert bin_action_value(-50.0, -50.0, +50.0, 128) == 0


def test_bin_action_value_last():
    assert bin_action_value(+50.0, -50.0, +50.0, 128) == 127


def test_bin_action_value_before():
    assert bin_action_value(-55.0, -50.0, +50.0, 128) == 0


def test_bin_action_value_after():
    assert bin_action_value(+55.0, -50.0, +50.0, 128) == 127

#
# unbin_action_value
#


def test_unbin_action_value_middle():
    assert round(unbin_action_value(105, -50.0, +50.0, 128)) == 33.0


def test_unbin_action_value_first():
    assert unbin_action_value(0, -50.0, +50.0, 128) == -50.0


def test_unbin_action_value_last():
    assert unbin_action_value(127, -50.0, +50.0, 128) == +50.0


def test_unbin_action_value_before():
    assert unbin_action_value(-3, -50.0, +50.0, 128) == -50.0


def test_unbin_action_value_after():
    assert unbin_action_value(+130, -50.0, +50.0, 128) == +50.0
