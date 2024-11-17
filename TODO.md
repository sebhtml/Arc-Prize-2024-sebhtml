# Sprint

- reduce number of train steps
- encode action in input using less tokens (rows, cols, value)
- reduce batch size
- move norm and dropout at same scope in transformer block
- increase dropout

- TODO test all actions in one batch
- print the output of nvidia-smi at the end of training
- print total wall-clock time for the training time

- use half precision float16
- implement rotations
- implement symmetries (flips)

# Backlog

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- add class add class Experience with (s, a, r, s')
