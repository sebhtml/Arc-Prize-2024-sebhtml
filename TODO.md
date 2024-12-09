# Research and Development

- move max_taken_actions_per_step to config
- unit tests for bin_action_value and unbin_action_value
- use target Q network to use Bellman equation
- remove class QLearningExample

# Backlog

- move print_current_state in GameState
- use GameState in Emulator
- use GameState instead of List[List[Cell]] everywhere

- implement epsilon-greedy policy for random exploration

- move functions that load json files to puzzle_data.py
- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- use half precision float16
