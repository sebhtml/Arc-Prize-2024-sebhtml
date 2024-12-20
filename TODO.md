# Research and Development

- add actions left (10), up (11), right (12), down (13)

- move training of action value network to agent.py
- return only max_action_value in select_action_with_target_action_value_network
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
- use select_action_with_policy