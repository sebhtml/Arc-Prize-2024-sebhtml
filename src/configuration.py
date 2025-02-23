
class Configuration:
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

    kaggle_input_path = "/workspace/kaggle-input/arc-prize-2024"
    logs_path = "/workspace/logs"
    episodes_dir_path = "/workspace/episodes"

    time_marker = '2024-11-09T00:18:25.063611+00:00'

    device = "cuda"

    #
    # Infrastructure configuration
    #
    api_key_file = "/workspace/runpod_api_key.yml"
    terminate_pod_at_the_end = False  # prod: True, dev: False

    #
    # Puzzle configuration
    #

    # See https://arcprize.org/play?task=3aa6fb7a
    selected_puzzle_id = "3aa6fb7a"

    #
    # Vision configuration
    #
    padding_char = ' '
    num_visual_fixations = 5
    visual_fixation_height = 5
    visual_fixation_width = 5
    # Each cell has one color and there are 10 colors.
    cell_value_size = 10

    #
    # Neural network model configuration
    #
    # Multiple of 4 for NVIDIA cublas WMMA
    # See https://docs.nvidia.com/cuda/cublas/#cublasltmatmul-regular-imma-conditions
    context_size = 196
    # Hidden size
    d_model = 384
    # Feed-forward size in transformer block
    d_ff = 1024
    num_actions = 10
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
    embedding_dropout = 0.0
    attention_head_dropout = 0.0
    attention_sublayer_dropout = 0.0
    ffn_sublayer_dropout = 0.0
    output_dropout = 0.0

    #
    # Training parameters
    #

    # See: A Recipe for Training Neural Networks
    # http://karpathy.github.io/2019/04/25/recipe/

    max_episodes = 1000
    min_percentage_correct_cells = 1.0
    max_winning_streak_length = 32
    target_network_update_period = 1000  # prod: 1000, dev: 100
    render_recorded_episodes = True

    verbose_advantage = False
    verbose_target_action_value_network = False
    use_action_value_network = False
    use_policy_network = True
    # "greedy" or "stochastic"
    policy_action_selection = "stochastic"
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
    batch_size = 32
    lr = 0.0001
    # In "Llama 2: Open Foundation and Fine-Tuned Chat Models" https://arxiv.org/abs/2307.09288, they use a weight decay of 0.1
    weight_decay = 0.1

    #
    # Q-learning parameters
    #
    discount = 0.99
    minimum_action_value = 0
    maximum_action_value = 49 * 1

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

    # File paths
    models_path = "/workspace/models"
    model_file_path = f"{models_path}/{time_marker}-{max_episodes}-q-network.pth"

    def __init__(self):
        for key, value in dict(Configuration.__dict__).items():
            if not key.startswith('_'):
                setattr(self, key, value)
