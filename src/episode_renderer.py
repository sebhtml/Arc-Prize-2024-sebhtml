from typing import List
from datetime import datetime, timezone
from typing import IO
from q_learning import GameState
from context import VACANT_CELL_CHAR
from vision import VACANT_CELL_VALUE, Cell


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
            print_game_state_with_colors(state, io,)

    io.close()


def print_cell(value: int, io: IO[str]):
    colors = {
        "0": "40m",  # black
        "1": "44m",  # blue
        "2": "41m",  # red
        "3": "42m",  # green
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


def print_state_with_colors(state: List[List[Cell]], io: IO[str]):
    example_input_height = len(state)
    example_input_width = len(state[0])

    for y in range(example_input_height):
        for x in range(example_input_width):
            cell_value = state[y][x].cell_value()
            print_cell(cell_value, io)
        io.write("\n")
    io.write("\n")


def print_game_state_with_colors(state: GameState, io: IO[str]):
    # Render example input.
    io.write("EXAMPLE_INPUT")
    io.write("\n")
    example_input = state.example_input().cells()
    print_state_with_colors(example_input, io,)

    # Render current state.
    io.write("CURRENT_STATE")
    io.write("\n")
    current_state = state.current_state()
    print_state_with_colors(current_state, io,)

    io.flush()
