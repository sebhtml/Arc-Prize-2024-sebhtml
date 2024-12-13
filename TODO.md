# Research and Development

- remove get_action_value_min_max
- remove sum_of_future_rewards
- remove unused arguments in all modules

- use mean action value from distributional action values, see https://arxiv.org/pdf/1710.02298 and https://arxiv.org/abs/1707.06887

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
