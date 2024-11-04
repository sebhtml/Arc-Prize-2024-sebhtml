# Sprint

- use runpod API to terminate pod after training if terminate_pod_after_training is True
  see https://github.com/runpod/runpod-python?tab=readme-ov-file#endpoints

- implement translations
- implement rotations
- implement symmetries (flips)

# Backlog

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- use "Feed forward mechanisms" from xformers
- use "Residual paths" from xformers
- add class add class Experience with (s, a, r, s')
