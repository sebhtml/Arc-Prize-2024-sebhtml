# Research and Development
- train after every taken action
- train once experience data set contains 4 * batch_size
- add class for configuration

- move print_current_state in GameState
- use GameState in Emulator
- use GameState instead of List[List[Cell]] everywhere

# Target Q Network

- make the QDN have only 10 actions: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- use a target Q network for the future rewards
- implement epsilon-greedy policy for random exploration
- also mask non-vacant cells in visual fixation

# Backlog


- remove type Cell

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