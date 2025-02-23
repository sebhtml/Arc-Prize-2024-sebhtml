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
from agent import test_policy, Agent
from environment import Environment
from training import train_model_using_experience_replay
from configuration import Configuration
from episode_renderer import render_episodes
from q_learning import ExampleInput
from vision import Cell

config = Configuration()


def get_puzzle_solution(venue, puzzle_id):
    solutions_file = f"{config.kaggle_input_path}/arc-agi_{venue}_solutions.json"
    f = open(solutions_file)
    solutions_data = json.load(f)
    solution = solutions_data[puzzle_id][0]
    return solution


def make_celled_state(state: List[List[int]]) -> List[List[Cell]]:
    celled_state = []
    for row in state:
        celled_row = []
        for value in row:
            celled_value = Cell(value)
            celled_row.append(celled_value)
        celled_state.append(celled_row)
    return celled_state


def make_celled_example(example: Tuple[List[List[int]], List[List[int]]]) -> Tuple[ExampleInput, List[List[Cell]]]:
    example_input, example_output = example
    celled_example_input = make_celled_state(example_input)
    celled_example_output = make_celled_state(example_output)
    celled_example_input = ExampleInput(celled_example_input)
    celled_example = (celled_example_input, celled_example_output)
    return celled_example


def load_puzzle_examples(puzzle_id: str, example_type: str) -> List[Tuple[ExampleInput, List[List[Cell]]]]:
    """
    - example_type is "train" or "test"
    Note that for the "test" venue, no solutions are provided.
    """

    venues = [
        "training",
        "evaluation",
        "test",
    ]

    venue = None
    puzzle_challenges_data = None

    for tentative_venue in venues:
        venue = tentative_venue
        challenges_file = f"{config.kaggle_input_path}/arc-agi_{venue}_challenges.json"
        f = open(challenges_file)
        challenges_data = json.load(f)
        try:
            puzzle_challenges_data = challenges_data[puzzle_id]
            break
        except:
            venue = None

    if puzzle_challenges_data == None:
        raise Exception(f"puzzle {puzzle_id} was not found.")

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

        celled_example = make_celled_example(example)

        puzzle_venue_examples.append(celled_example)
    return puzzle_venue_examples


def main():
    cell_value_size = config.cell_value_size
    device = torch.device(config.device,)
    environment = Environment(cell_value_size)
    agent = Agent(config, device)
    # RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
    # torch.compile does not work on the NVIDIA P100
    # torch.compile works on runpod.io with a NVIDIA A40
    # model = torch.compile(model)

    puzzle_train_examples = load_puzzle_examples(
        config.selected_puzzle_id, "train")

    print("puzzle_train_examples")
    print(len(puzzle_train_examples))

    puzzle_test_examples = load_puzzle_examples(
        config.selected_puzzle_id, "test")

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
            config.max_episodes,
        )

    if config.save_neural_net_model:
        torch.save(agent.policy_network().state_dict(),
                   config.model_file_path)

    # Check if the auto-regressive inference AI is able to predict the output for the train examples.
    if config.run_autoregressive_inference_on_train_examples:
        test_policy(
            puzzle_train_examples, agent, config, config.cell_value_size, config.context_size, config.batch_size, device, environment,)

    # Check if the auto-regressive inference AI is able to predict the output for the test example.
    if config.run_autoregressive_inference_on_test_examples:
        test_policy(
            puzzle_test_examples, agent, config, config.cell_value_size, config.context_size, config.batch_size, device, environment,)

    # Render recorded episodes.
    if config.render_recorded_episodes:
        episodes = environment.recorded_episodes()
        render_episodes(config.selected_puzzle_id,
                        episodes, config.episodes_dir_path,)

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
