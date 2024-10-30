# Sprint

- Increase model and lower batch size since GPU VRAM is underused: allocated: 36582MiB / 46068MiB

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)

# Backlog

- add class QLearningState
- add class QLearningActionValue

- implement translations
- implement rotations

- use "Feed forward mechanisms" from xformers
- use "Residual paths" from xformers
- add class add class Experience with (s, a, r, s')
