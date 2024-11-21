# Sprint

- don't allow token OUTSIDE_BOARD_CHAR in input of neural network.
- implement symmetries (flips)
- implement rotations
- use only one learned embedding
- don't talk about playout
- don't count cells outside of the grid in the future rewards.
- don't assume perfect play to compute future rewards
- encode action in input using less tokens (rows, cols, value)
VACANT_CELL_CHAR = '_'
AROUND_ACTION_CHAR = ','
OUTSIDE_BOARD_CHAR = '.'
OUTSIDE_BOARD_CELL_VALUE = -1

# Backlog

- move functions that load json files to puzzle_data.py
- use another character for "outside of field of view"
- add class PuzzleState
- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- reduce batch size

- use half precision float16

- add class QLearningState
- add class QLearningActionValue

- add class add class Experience with (s, a, r, s')
