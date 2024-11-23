# Sprint

- use VACANT_CELL_CHAR = '_'
- use AROUND_ACTION_CHAR = ','
- use OUTSIDE_BOARD_CHAR = '.'
- use OUTSIDE_BOARD_CELL_VALUE = -1
- implement OUTSIDE_BOARD_CHAR
- don't use OUTSIDE_BOARD_CHAR in input to compute action value

- use str instead of int for cell values
- implement symmetries (flips)
- implement rotations
- use only one learned embedding
- don't talk about playout
- don't count cells outside of the grid in the future rewards.
- don't assume perfect play to compute future rewards
- don't pass the AROUND_ACTION_CHAR chars to the neural network
- Verify if the HDF5 schema must be changed.

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
