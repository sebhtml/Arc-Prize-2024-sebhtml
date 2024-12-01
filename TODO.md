# Research and Development Objective

- make the QDN have only 10 actions: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

# Backlog

- use a target Q network for the future rewards

- remove total_train_examples
- add class for configuration

- generate_train_action_examples should receive just one puzzle example
- play a bunch of games, then extract examples

- add class StateActionExample

- use AROUND_ACTION_CHAR = ','
- don't pass the AROUND_ACTION_CHAR chars to the neural network

- move functions that load json files to puzzle_data.py
- move norm and dropout at same scope in transformer block
- increase dropout

- don't assume perfect play to compute future rewards
- reduce number of train steps
- reduce batch size
- use half precision float16

- make the QDN have only 14 actions: left, up, down, right, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9