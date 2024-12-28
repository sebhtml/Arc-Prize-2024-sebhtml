from typing import List
from asciimatics.screen import Screen
from asciimatics.renderers import FigletText, StaticRenderer
import time
import os
from datetime import datetime, timezone
import cv2

from q_learning import GameState


def render_state(state: GameState, screen: Screen,):
    # Render example input.
    example_input = state.example_input()
    example_input_height = len(example_input)
    example_input_width = len(example_input[0])

    for x in range(example_input_width):
        for y in range(example_input_height):
            text = example_input[y][x].value()
            screen.print_at(text, x, y,)

    # Render current state.
    current_state_x_offset = example_input_width + 4
    current_state = state.current_state()
    current_state_height = len(current_state)
    current_state_width = len(current_state[0])

    for x in range(current_state_width):
        for y in range(current_state_height):
            text = current_state[y][x].value()
            x_with_offset = x + current_state_x_offset
            screen.print_at(text, x_with_offset, y,)


def render_episodes(puzzle_id: str, episodes: List[List[GameState]], video_dir_path: str,):
    """
    Render puzzle episodes as a video file.
    """

    dynamic_time_marker = datetime.now(timezone.utc).isoformat()
    video_file_path = f"{video_dir_path}/puzzle-{puzzle_id}-animation-{dynamic_time_marker.replace(':', '-')}.txt"
    screen = Screen.open()
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(video_file_path, fourcc,
    #  1.0, (screen.width, screen.height))
    #out = open(video_file_path, "wb")

    for episode in episodes:
        for state in episode:
            screen.clear()
            render_state(state, screen,)
            # frame = screen.get_image()
            # out.write(frame)
            screen.refresh()
            #frame = screen._buffer.plain_image
            #out.write(bytes(frame))
            #out.write("\n\n")
            time.sleep(0.1)

    # out.release()
    #out.close()
    screen.close()
