# Research and Development

- remove function mask_cells

- add a policy network that uses a off-policy value network to compute the advantage (basically actor critic)
    see Off-Policy Actor-Critic https://icml.cc/2012/papers/268.pdf
    see Actor-Critic Algorithms https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- use argmax in select_action_with_deep_q_network
- add old policy network to use PPO (proximal policy optimization)
    see Off-Policy Proximal Policy Optimization
        https://ojs.aaai.org/index.php/AAAI/article/view/26099

# Backlog

- remove unused arguments in all modules
- move print_current_state in GameState
- use GameState in Environment
- use GameState instead of List[List[Cell]] everywhere

- implement epsilon-greedy policy for random exploration

- move functions that load json files to puzzle_data.py
- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- use half precision float16
