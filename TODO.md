# Sprint

- generate_train_action_examples should receive just one puzzle example
- play a bunch of games, then extract examples
- interleave playing and learning using the ReplayBuffer

- add class StateActionExample

- use AROUND_ACTION_CHAR = ','
- don't pass the AROUND_ACTION_CHAR chars to the neural network

# Backlog

- move functions that load json files to puzzle_data.py
- move norm and dropout at same scope in transformer block
- increase dropout

- don't assume perfect play to compute future rewards
- reduce number of train steps
- reduce batch size
- use half precision float16
