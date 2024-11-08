# Sprint

- increase batch_size and train_samples since we are now using only 26 GB / 46 GB VRAM.
- use half precision float16
- implement rotations
- implement symmetries (flips)

# Backlog

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- add class add class Experience with (s, a, r, s')
