# Research and Development

- calculate salient cells once at the beginningv
- num_visual_fixations should be 5 actually
- clean up imports
- rename module q_learning
- test with 009d5c81

# Backlog

- process all experience in training loop because right now with reinforce, I get this:
logits tensor([[11.0497, -1.1402, -1.0108, -1.6909, -1.8992, -1.2135, -1.2171, -1.5647,
         -0.8588, -1.3075],
- use term "returns" not "rewards"

- retry reinforce algorithm
- remove get_correct

- normalize rewards
- move cell_match_reward to config
- move cell_mismatch_reward to config

- to select visual fixations, use sampling in training, argmax in test)




- add puzzle id in png file names

- use Dropout in classifier of policy network model
- move norm and dropout at same scope in transformer block

- remove vacant variables
- use new puzzle

- for data augmentation, maybe remove random cols or rows
- remove QDN usage
- use BERT_large parameters L=12, H=768, A=12

# Vision

- compute saliency with GPU
- visual_fixation_width: int, visual_fixation_height: int, vs convolution kernel size

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

- move training of action value network in a step method in Agent
- add actions left (10), up (11), right (12), down (13)
- return only max_action_value in select_action_with_target_action_value_network

- move training of action value network to agent.py

# Policy gradient methods

- add old policy network to use PPO (proximal policy optimization)
    see Off-Policy Proximal Policy Optimization
        https://ojs.aaai.org/index.php/AAAI/article/view/26099

# Reinforcement learning

