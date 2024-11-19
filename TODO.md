# Sprint

- implement symmetries (flips)

- implement rotations
- use only one learned embedding

# Backlog

- move functions that load json files to puzzle_data.py
- encode action in input using less tokens (rows, cols, value)
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
