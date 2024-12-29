from typing import List
from datetime import datetime, timezone
from typing import IO
from q_learning import GameState, VACANT_CELL_VALUE
from context import VACANT_CELL_CHAR


def render_episodes(puzzle_id: str, episodes: List[List[GameState]], episodes_dir_path: str,):
    """
    Render puzzle episodes as a video file.
    """

    dynamic_time_marker = datetime.now(timezone.utc).isoformat()
    episodes_file_path = f"{episodes_dir_path}/puzzle-{puzzle_id}-episodes-{dynamic_time_marker.replace(':', '-')}.txt"
    io = open(episodes_file_path, "w")

    for i, episode in enumerate(episodes):
        for j, state in enumerate(episode):
            io.write(
                f"Episodes: {len(episodes)},  Episode: {i},  State: {j}\n")
            print_with_colors(state, io,)

    io.close()


def print_cell(value: int, io: IO[str]):
    colors = {
        "0": "40m",  # black
        "1": "44m",  # blue
        "2": "41m",  # red
        "4": "43m",  # yellow
        "6": "45m",  # purple
        "7": "48;5;208m",  # orange
        "8": "46m",  # cyan
    }

    char = VACANT_CELL_CHAR if value == VACANT_CELL_VALUE else str(value)

    before = ""
    after = ""
    try:
        color = colors[char]
        before = f"\033[{color}"
        after = f"\033[0m"
    except:
        pass

    io.write(f"{before} {char} {after}")


def print_with_colors(state: GameState, io: IO[str]):
    # Render example input.
    io.write("EXAMPLE_INPUT")
    io.write("\n")
    example_input = state.example_input()
    example_input_height = len(example_input)
    example_input_width = len(example_input[0])

    for y in range(example_input_height):
        for x in range(example_input_width):
            cell_value = example_input[y][x].value()
            print_cell(cell_value, io)
        io.write("\n")
    io.write("\n")

    # Render current state.
    io.write("CURRENT_STATE")
    io.write("\n")
    current_state = state.current_state()
    current_state_height = len(current_state)
    current_state_width = len(current_state[0])

    for y in range(current_state_height):
        for x in range(current_state_width):
            cell_value = current_state[y][x].value()
            print_cell(cell_value, io)
        io.write("\n")
    io.write("\n")

    io.flush()
