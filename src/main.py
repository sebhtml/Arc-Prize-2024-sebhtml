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


# import torch_xla
# import torch_xla.core.xla_model as xm
from typing import List, Tuple
import subprocess
import time
import torch
import json
from model import DecoderOnlyTransformerModel
from infrastructure import terminate_pod
from agent import apply_puzzle_action_value_policy
from training import train_model_using_experience_replay
from configuration import Configuration

config = Configuration()

# device = xm.xla_device()
# device = torch.device("cpu")


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
    device = torch.device("cuda")
    action_value_network = DecoderOnlyTransformerModel(
        config.vocab_size, config.d_model, config.d_ff,
        config.input_dropout, config.attention_head_dropout, config.attention_sublayer_dropout, config.ffn_sublayer_dropout,
        config.num_heads, config.context_size, config.num_layers, config.num_actions, config.num_classes, device)
    # RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
    # torch.compile does not work on the NVIDIA P100
    # torch.compile works on runpod.io with a NVIDIA A40
    # model = torch.compile(model)
    action_value_network.to(device)

    puzzle_train_examples = load_puzzle_examples(
        "training", config.selected_puzzle_id, "train")

    print("puzzle_train_examples")
    print(len(puzzle_train_examples))

    puzzle_test_examples = load_puzzle_examples(
        "training", config.selected_puzzle_id, "test")

    model_total_params = sum(p.numel()
                             for p in action_value_network.parameters())
    print(f"parameters: {model_total_params}")

    if config.load_model:
        print("Loading model")
        state_dict = torch.load(config.model_file_path, weights_only=True)
        action_value_network.load_state_dict(state_dict)

    if config.train_model:
        train_model_using_experience_replay(
            config,
            config.context_size, config.batch_size, device, action_value_network, config.total_train_examples,
            puzzle_train_examples, config.cell_value_size,
            config.discount, config.padding_char, config.num_classes, config.shuffle_train_examples, config.lr,
            config.weight_decay, config.max_grad_norm, config.print_model_outputs, config.save_step_losses,
            config.num_steps,
        )

    if config.save_neural_net_model:
        torch.save(action_value_network.state_dict(),
                   config.model_file_path)

    # Check if the auto-regressive inference AI is able to predict the output for the train examples.
    if config.run_autoregressive_inference_on_train_examples:
        apply_puzzle_action_value_policy(
            puzzle_train_examples, action_value_network, config.padding_char, config.cell_value_size, config.context_size, config.batch_size, device)

    # Check if the auto-regressive inference AI is able to predict the output for the test example.
    if config.run_autoregressive_inference_on_test_examples:
        apply_puzzle_action_value_policy(
            puzzle_test_examples, action_value_network, config.padding_char, config.cell_value_size, config.context_size, config.batch_size, device)

    if config.terminate_pod_at_the_end:
        terminate_pod(config.api_key_file)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("--- TOTAL_TIME: %s seconds ---" % (time.time() - start_time))
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode(encoding='utf-8')
    print(output)
