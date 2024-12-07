# Research and Development

- use 1 Linear in SwiGLU and reshape it in (2, dim)
- produce 10 action values with neural network with 1 Linear 
self.linear = nn.Linear(input_dim, 2*input_dim)
chunks = x_linear_pos, x_linear_neg = x_linear.chunk(2, dim=1)
- make the QDN have only 10 actions: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

- also mask non-vacant cells in visual fixation
- use Bellman equation

- add class for configuration

- move print_current_state in GameState
- use GameState in Emulator
- use GameState instead of List[List[Cell]] everywhere

# Target Q Network


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
