# Author:
# Sebastien Boisvert <sebhtml@protonmail.com>
#
#

# Hardware used:

# Legion
# - NVIDIA GeForce RTX 4060 8188MiB

# Kaggle
# - GPU: NVIDIA P100

# Runpod
# - NVIDIA A40 48 GB VRAM
# - NVIDIA RTX A4000 16 GB VRAM

# os.system("pip uninstall fastai torchvision torchaudio")  # nopep8
# For TPUs # nopep8
# os.system(   # nopep8
#    "pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html")   # nopep8
# os.system("pip install xformers")  # nopep8


from typing import List, Tuple
import subprocess
import time
import torch
import json
from model import ActionValueNetworkModel
from infrastructure import terminate_pod
from agent import apply_policy_network, Agent
from environment import Environment
from training import train_model_using_experience_replay
from configuration import Configuration
from video_renderer import render_episodes

config = Configuration()


def get_puzzle_solution(venue, puzzle_id):
    solutions_file = f"{config.kaggle_input_path}/arc-agi_{venue}_solutions.json"
    f = open(solutions_file)
    solutions_data = json.load(f)
    solution = solutions_data[puzzle_id][0]
    return solution


def load_puzzle_examples(venue, puzzle_id, example_type) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """
    - venue is "training" or "evaluation" or "test"
    - example_type is "train" or "test"
    Note that for the "test" venue, no solutions are provided.
    """
    challenges_file = f"{config.kaggle_input_path}/arc-agi_{venue}_challenges.json"
    f = open(challenges_file)
    challenges_data = json.load(f)
    puzzle_challenges_data = challenges_data[puzzle_id]
    puzzle_examples = puzzle_challenges_data[example_type]

    puzzle_venue_examples = []
    for puzzle_example in puzzle_examples:
        example_input = puzzle_example["input"]
        example_output = None
        if venue == "test":
            pass
        elif example_type == "train":
            example_output = puzzle_example["output"]
        else:
            example_output = get_puzzle_solution(venue, puzzle_id)

        example = (example_input, example_output)
        puzzle_venue_examples.append(example)
    return puzzle_venue_examples


def main():
    cell_value_size = config.cell_value_size
    device = torch.device("cuda")
    environment = Environment(cell_value_size)
    agent = Agent(config, device)
    # RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
    # torch.compile does not work on the NVIDIA P100
    # torch.compile works on runpod.io with a NVIDIA A40
    # model = torch.compile(model)

    puzzle_train_examples = load_puzzle_examples(
        "training", config.selected_puzzle_id, "train")

    print("puzzle_train_examples")
    print(len(puzzle_train_examples))

    puzzle_test_examples = load_puzzle_examples(
        "training", config.selected_puzzle_id, "test")

    model_total_params = sum(p.numel()
                             for p in agent.policy_network().parameters())
    print(f"parameters: {model_total_params}")

    if config.load_model:
        print("Loading model")
        state_dict = torch.load(config.model_file_path, weights_only=True)
        agent.policy_network().load_state_dict(state_dict)

    if config.train_model:
        train_model_using_experience_replay(
            environment,
            config,
            config.context_size, config.batch_size, device, agent,
            puzzle_train_examples, config.cell_value_size,
            config.discount, config.padding_char, config.num_classes, config.shuffle_train_examples, config.lr,
            config.weight_decay, config.max_grad_norm, config.print_model_outputs, config.save_step_losses,
            config.num_steps,
        )

    if config.save_neural_net_model:
        torch.save(agent.policy_network().state_dict(),
                   config.model_file_path)

    # Check if the auto-regressive inference AI is able to predict the output for the train examples.
    if config.run_autoregressive_inference_on_train_examples:
        apply_policy_network(
            puzzle_train_examples, agent, config.padding_char, config.cell_value_size, config.context_size, config.batch_size, device, environment,)

    # Check if the auto-regressive inference AI is able to predict the output for the test example.
    if config.run_autoregressive_inference_on_test_examples:
        apply_policy_network(
            puzzle_test_examples, agent, config.padding_char, config.cell_value_size, config.context_size, config.batch_size, device, environment,)

    # Render recorded episodes.
    if config.render_recorded_episodes:
        episodes = environment.recorded_episodes()
        render_episodes(config.selected_puzzle_id,
                        episodes, config.video_dir_path,)

    if config.terminate_pod_at_the_end:
        terminate_pod(config.api_key_file)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = time.time() - start_time
    print(f"--- TOTAL_TIME: {seconds} seconds ---")
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode(encoding='utf-8')
    print(output)
