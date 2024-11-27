# Sprint

- move translate_board to vision.py
- move do_visual_fixation to vision.py
- rename samples to examples
- implement symmetries (flips)

- use AROUND_ACTION_CHAR = ','
- don't pass the AROUND_ACTION_CHAR chars to the neural network
- don't assume perfect play to compute future rewards
- don't talk about playout

# Backlog

- move functions that load json files to puzzle_data.py
- add class PuzzleState
- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- reduce batch size

- use half precision float16

- add class QLearningState
- add class QLearningActionValue

