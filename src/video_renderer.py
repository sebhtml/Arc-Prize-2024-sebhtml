from typing import List
import time
from datetime import datetime, timezone
from q_learning import GameState
from report import print_with_colors


def render_episodes(puzzle_id: str, episodes: List[List[GameState]], episodes_dir_path: str,):
    """
    Render puzzle episodes as a video file.
    """

    dynamic_time_marker = datetime.now(timezone.utc).isoformat()
    episodes_file_path = f"{episodes_dir_path}/puzzle-{puzzle_id}-episodes-{dynamic_time_marker.replace(':', '-')}.txt"
    io = open(episodes_file_path, "w")

    for i, episode in enumerate(episodes):
        for j, state in enumerate(episode):
            io.write(f"Episodes: {len(episodes)},  Episode: {i},  State: {j}\n")
            print_with_colors(state, io,)

    io.close()
