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
import subprocess
import time
from datetime import datetime, timezone
import sys
import torch
import json
import pandas as pd
from model import DecoderOnlyTransformerModel
from infrastructure import terminate_pod
from report import plot_train_loss_graph
from agent import apply_puzzle_action_value_policy, generate_examples
from training import MyDataset, train, print_model_outputs_for_train_examples

device = torch.device("cuda")

# device = xm.xla_device()
# device = torch.device("cpu")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#
# /kaggle/input/arc-prize-2024/arc-agi_training_challenges.json
# /kaggle/input/arc-prize-2024/arc-agi_training_solutions.json
#
# /kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json
# /kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json
#
# /kaggle/input/arc-prize-2024/arc-agi_test_challenges.json
#
# /kaggle/input/arc-prize-2024/sample_submission.json

# Paths
# On Kaggle
# kaggle_input_path = "/kaggle/input/arc-prize-2024"
# logs_path = "/workspace/logs"
# On runpod

kaggle_input_path = "/workspace/kaggle-input"
logs_path = "/workspace/logs"

time_marker = '2024-11-09T00:18:25.063611+00:00'
dynamic_time_marker = datetime.now(timezone.utc).isoformat()
train_loss_csv_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.csv"
train_loss_png_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.png"

#
# Infrastructure configuration
#
api_key_file = "/workspace/runpod_api_key.yml"
terminate_pod_at_the_end = False

#
# Puzzle configuration
#

# See https://arcprize.org/play?task=3aa6fb7a
selected_puzzle_id = "3aa6fb7a"

# Each cell has one color and there are 10 colors.
cell_value_size = 10

#
# Playout simulation configuration
#

generate_train_examples = True
# Use 100000 for dev, and use 10000000 for training the model.
total_train_examples = 100000
stop_after_generating_examples = False
# Since we use the model itself to generate games, we can not use more than one CPU
playout_simulation_cpu_count = 1
train_dataset_path = f"/workspace/train_datasets/{time_marker}-{selected_puzzle_id}-{total_train_examples}.hdf5"
discount = 0.99
padding_char = ' '

#
# Neural network model configuration
#


models_path = "/workspace/models"
model_file_path = f"{models_path}/{time_marker}-{total_train_examples}-q-network.pth"
# Multiple of 4 for NVIDIA cublas WMMA
# See https://docs.nvidia.com/cuda/cublas/#cublasltmatmul-regular-imma-conditions
context_size = 148
# Hidden size
d_model = 384
# Feed-forward size in transformer block
d_ff = 1024
num_classes = 128
num_heads = 8
num_layers = 8
vocab_size = 128

#
# Dropout regularization
#

# See
# Improving neural networks by preventing co-adaptation of feature detectors
# https://arxiv.org/pdf/1207.0580
# See:
# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
# https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
input_dropout = 0.2
attention_head_dropout = 0.2
attention_sublayer_dropout = 0.2
ffn_sublayer_dropout = 0.1

#
# Training parameters
#

# See: A Recipe for Training Neural Networks
# http://karpathy.github.io/2019/04/25/recipe/

shuffle_train_examples = True
# In "Llama 2: Open Foundation and Fine-Tuned Chat Models" https://arxiv.org/abs/2307.09288, they do gradient clipping with norm=1.0
max_grad_norm = 1.0

# For batch_size:
# - 8192 for the TPU machine since "TPU-v3-8" has 330 GB RAM
# - 512 for TPU-v3-8 with 1 TPU
# - 512 for the NVIDIA P100 GPU since "P100" has 16 GB VRAM
# - 1024 for CPU since "CPU" has 29 GB RAM
# On runpod:
# - 1536 with NVIDIA A40 (48 GB VRAM)
# - 512 with NVIDIA A4000 (16 GB VRAM)
# See https://x.com/ylecun/status/989610208497360896?lang=en
batch_size = 512
lr = 0.0001
# In "Llama 2: Open Foundation and Fine-Tuned Chat Models" https://arxiv.org/abs/2307.09288, they use a weight decay of 0.1
# In "Grandmaster-Level Chess Without Search" https://arxiv.org/html/2402.04494v1, they don't say what weight decay they used.
weight_decay = 0.1

# Use 1 epoch when training the model, 4 for dev
num_epochs = 1

#
# Options for loading AI neural net model
#
load_model = False
#
# Options for training AI neural net model
#
train_model = True
save_step_losses = True
save_neural_net_model = True
#
# Options for evaluating AI neural net model
#
print_model_outputs = True
run_autoregressive_inference_on_train_examples = True
run_autoregressive_inference_on_test_examples = True


def get_puzzle_solution(venue, puzzle_id):
    solutions_file = f"{kaggle_input_path}/arc-agi_{venue}_solutions.json"
    f = open(solutions_file)
    solutions_data = json.load(f)
    solution = solutions_data[puzzle_id][0]
    return solution


def load_puzzle_examples(venue, puzzle_id, example_type):
    """
    - venue is "training" or "evaluation" or "test"
    - example_type is "train" or "test"
    Note that for the "test" venue, no solutions are provided.
    """
    challenges_file = f"{kaggle_input_path}/arc-agi_{venue}_challenges.json"
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
    model = DecoderOnlyTransformerModel(
        vocab_size, d_model, d_ff,
        input_dropout, attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
        num_heads, context_size, num_layers, num_classes, device)
    # RuntimeError: Found Tesla P100-PCIE-16GB which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.0
    # torch.compile does not work on the NVIDIA P100
    # torch.compile works on runpod.io with a NVIDIA A40
    # model = torch.compile(model)
    model.to(device)

    puzzle_train_examples = load_puzzle_examples(
        "training", selected_puzzle_id, "train")

    print("puzzle_train_examples")
    print(len(puzzle_train_examples))

    puzzle_test_examples = load_puzzle_examples(
        "training", selected_puzzle_id, "test")

    if generate_train_examples:
        generate_examples(train_dataset_path, total_train_examples, puzzle_train_examples, cell_value_size,
                          discount, padding_char, playout_simulation_cpu_count)

        if stop_after_generating_examples:
            sys.exit(0)

    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {model_total_params}")

    if load_model:
        print("Loading model")
        state_dict = torch.load(model_file_path, weights_only=True)
        model.load_state_dict(state_dict)

    if train_model:
        print("Training model")
        # Create a dataset.
        dataset = MyDataset(train_dataset_path, context_size, num_classes,)

        step = 0
        step, steps, losses = train(
            dataset, batch_size, shuffle_train_examples, step, model,
            num_epochs, lr, weight_decay, max_grad_norm, device,)

        if print_model_outputs:
            print_model_outputs_for_train_examples(
                dataset, batch_size, model, device,)

        if save_step_losses:
            df = pd.DataFrame(data={'step': steps, 'loss': losses})
            df.to_csv(train_loss_csv_path, index=False)
            plot_train_loss_graph(steps, losses, train_loss_png_path)

        if save_neural_net_model:
            torch.save(model.state_dict(),
                       model_file_path)

    # Check if the auto-regressive inference AI is able to predict the output for the train examples.
    if run_autoregressive_inference_on_train_examples:
        apply_puzzle_action_value_policy(
            puzzle_train_examples, model, padding_char, cell_value_size, context_size, batch_size, device)

    # Check if the auto-regressive inference AI is able to predict the output for the test example.
    if run_autoregressive_inference_on_test_examples:
        apply_puzzle_action_value_policy(
            puzzle_test_examples, model, padding_char, cell_value_size, context_size, batch_size, device)

    if terminate_pod_at_the_end:
        terminate_pod(api_key_file)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("--- TOTAL_TIME: %s seconds ---" % (time.time() - start_time))
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    output = result.stdout.decode(encoding='utf-8')
    print(output)
