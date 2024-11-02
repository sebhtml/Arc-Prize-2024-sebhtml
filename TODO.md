# Sprint

- Increase model size since GPU VRAM is underused: allocated: 36582MiB / 46068MiB
- solve this error:
BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')

# Backlog

- implement translations

- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- add class QLearningState
- add class QLearningActionValue

- implement rotations

- use "Feed forward mechanisms" from xformers
- use "Residual paths" from xformers
- add class add class Experience with (s, a, r, s')
