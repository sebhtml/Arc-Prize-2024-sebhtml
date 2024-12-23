# Research and Development

- don't mask cells
- move log_softmax outside of model
- move training of action value network in a step method in Agent
- remove total_train_examples
- move cell_match_reward to config
- move cell_mismatch_reward to config
- add actions left (10), up (11), right (12), down (13)

- move training of action value network to agent.py

# Refactoring

- move make_example_tensor to Context
- move tokenize_example_input to Context
- move tokens_to_text to Context

- remove unused arguments in all modules
- move print_current_state in GameState
- use GameState in Environment
- use GameState instead of List[List[Cell]] everywhere

- move functions that load json files to puzzle_data.py

# Value network

- use load_dict for target net
- implement epsilon-greedy policy for random exploration

# Model

- move norm and dropout at same scope in transformer block
- increase dropout

- reduce number of train steps
- use half precision float16

# Policy gradient methods

- use select_action_with_policy
- return only max_action_value in select_action_with_target_action_value_network
- add old policy network to use PPO (proximal policy optimization)
    see Off-Policy Proximal Policy Optimization
        https://ojs.aaai.org/index.php/AAAI/article/view/26099
