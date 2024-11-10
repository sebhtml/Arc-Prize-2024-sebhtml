# Sprint

- use CustomAttention
- pass only x to CustomAttention instead of q,k,v
- fix dropout position in FeedForward class
- increase batch_size and train_samples since we are now using only 28846MiB / 46068MiB instead of 43039MiB /  46068MiB on the NVIDIA A40 GPU.
- encode action in input using less tokens (rows, cols, value)

- use half precision float16
- implement rotations
- implement symmetries (flips)

# Backlog

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- add class add class Experience with (s, a, r, s')
