# Sprint

- use AROUND_ACTION_CHAR = ','
- don't count cells outside of the grid in the future rewards.

- implement symmetries (flips)
- implement rotations
- use only one learned embedding
- don't talk about playout
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
