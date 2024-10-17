# Sprint

- check if the auto-regressive inference AI is able to predict the output for the train examples
 to check if model generalizes well for a known puzzle

- Increase model and lower batch size since GPU VRAM is underused: allocated: 36582MiB / 46068MiB

- find out why there are peaks in the loss curve here:
	https://docs.google.com/spreadsheets/d/19pwa_mYlXxRR5YxxytdO2z0Fz6SE-CwfJ9mc-I_TNLo/edit?gid=842182230#gid=842182230
- there are too many places where we manipulate 4 things for instances of SampleInputTokens (they have different learned embedding)
- do a code review of all the source code

# Backlog

- investigate model inference predicted action value using the function print_inferred_action_value.


- generate action_examples once and write them to disk

- add class QLearningState
- add class QLearningActionValue

- implement translations
- implement rotations

- use "Feed forward mechanisms" from xformers
- use "Residual paths" from xformers
- add class add class Experience with (s, a, r, s')

- check if the auto-regressive inference AI is able to predict the output for the test example.


