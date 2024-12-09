from training import bin_action_value

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
