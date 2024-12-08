# Research and Development

- make the QDN have only 10 actions: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

- use target Q network to use Bellman equation 

- add class for configuration

- move print_current_state in GameState
- use GameState in Emulator
- use GameState instead of List[List[Cell]] everywhere

- implement epsilon-greedy policy for random exploration

# Backlog

- remove type Cell

- add class StateActionExample

- use AROUND_ACTION_CHAR = ','
- don't pass the AROUND_ACTION_CHAR chars to the neural network

- move functions that load json files to puzzle_data.py
- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- use half precision float16
