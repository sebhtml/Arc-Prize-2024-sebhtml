# Sprint

- TODO test all actions in one batch
- print the output of nvidia-smi at the end of training
- print total wall-clock time for the training time
- move norm and dropout at same scope in transformer block
- use Flash attention
- train 16 models with batch_size= 32, and use average gradient
- increase model size since we are now using only 28846MiB / 46068MiB instead of 43039MiB /  46068MiB on the NVIDIA A40 GPU.
- try replacing SwiGLU with GELU
- encode action in input using less tokens (rows, cols, value)

- use half precision float16
- implement rotations
- implement symmetries (flips)

# Backlog

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- add class add class Experience with (s, a, r, s')
