import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime, timezone
from typing import List, Tuple
from agent import make_example_tensor, select_action_with_deep_q_network, play_game_using_model, Agent, unbin_action_value, bin_action_value
from context import tokens_to_text, tokenize_example_input
from report import plot_train_loss_graph, plot_total_rewards_graph
from model import ActionValueNetworkModel
from environment import Environment, generate_cell_actions
from vision import do_visual_fixation
from configuration import Configuration
from q_learning import Experience


def get_target_action_value(
        experience: Experience,
        padding_char: str,
        context_size: int,
        cell_value_size: int,
        minimum_action_value: float,
        maximum_action_value: float,
        num_classes: int,
        batch_size: int,
        discount: float,
        device: torch.device,
        target_action_value_network: ActionValueNetworkModel,
        verbose: bool,
) -> float:
    """
    This software used reinforcement learning.
    It uses Q-learning.

    See https://en.wikipedia.org/wiki/Q-learning
    See https://en.wikipedia.org/wiki/Bellman_equation

    See
    Dynamic Programming
    https://www.science.org/doi/10.1126/science.153.3731.34

    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    reward = experience.reward()

    example_input = experience.next_state().example_input()
    current_state = experience.next_state().current_state()

    candidate_actions = generate_cell_actions(
        current_state, cell_value_size)

    is_terminal = len(candidate_actions) == 0

    if is_terminal:
        return reward

    best_action, best_action_value, action_values = select_action_with_deep_q_network(
        example_input,
        current_state,
        candidate_actions,
        padding_char,
        context_size,
        batch_size,
        device,
        target_action_value_network,
        verbose,
    )

    best_action_value = unbin_action_value(
        best_action_value, minimum_action_value, maximum_action_value, num_classes)

    # We use the Bellman equation.
    # See https://en.wikipedia.org/wiki/Bellman_equation
    #
    # Given an experience
    # (x, a, r, x')
    # where
    #   x                        is the state
    #   a in Gamma(x)            is the action
    #   r = F(s, a)              is the reward
    #   x' = T(s, a)             is the new state
    #
    # We have
    #
    # V(x) = max_{a \in \Gamma(x)} [ F(x, a) + \beta * V(T(x, a)) ]
    #
    # Here, concretely, we have:
    #
    #
    #  Mathematical expression           Python expression
    #  -----------------------------------------------------
    #  a                                 experience.action()
    #  F(x, a)                           reward
    #  \beta                             discount
    #  V(T(x, a))                        best_action_value
    #

    # In Q-learning (https://en.wikipedia.org/wiki/Q-learning)
    # we write the Bellman equation as the following mathematical expresssion:
    #
    # We note the state with s instead of x.
    #
    # The discount is noted with gamma instead of beta.
    #
    # Q: S x A -> R
    #
    #
    # \hat{Q}(s, a) = r + gamma * max_{a' \in \Gamma(s')} \hat{Q}(s', a')
    #
    #
    # with
    # s' = T(s, a)
    # s \in S
    # a \in A
    # a \in \Gamma(s)

    return reward + discount * best_action_value


class MyDataset(Dataset):
    def __init__(self,
                 train_examples: List[Experience],
                 config: Configuration,
                 device: torch.device,
                 agent: Agent,
                 ):
        self.__train_examples = train_examples
        self.__config = config
        self.__device = device
        self.__agent = agent
        self.__printed = False

    def __len__(self):
        return len(self.__train_examples)

    def __getitem__(self, idx):
        experience = self.__train_examples[idx]
        example_input = experience.state().example_input()
        current_state = experience.state().current_state()
        candidate_action = experience.action()

        (attented_example_input, attented_current_state, attented_candidate_action,
         translation_x, translation_y) = do_visual_fixation(
            example_input, current_state, candidate_action)

        input_tokens = tokenize_example_input(
            current_state,
            attented_example_input, attented_current_state, self.__config.padding_char)

        if not self.__printed:
            print("input_text")
            print(tokens_to_text(input_tokens))
            self.__printed = True

        item_input = make_example_tensor(
            input_tokens, self.__config.context_size)

        action_index = torch.tensor(experience.action().cell_value())

        verbose = self.__config.verbose_target_action_value_network
        action_value = get_target_action_value(
            experience,
            self.__config.padding_char,
            self.__config.context_size,
            self.__config.cell_value_size,
            self.__config.minimum_action_value,
            self.__config.maximum_action_value,
            self.__config.num_classes,
            self.__config.batch_size,
            self.__config.discount,
            self.__device,
            self.__agent.target_action_value_network(),
            verbose,
        )
        action_value_bin = bin_action_value(
            action_value, self.__config.minimum_action_value, self.__config.maximum_action_value, self.__config.num_classes)
        action_value_bin = torch.tensor(action_value_bin)

        item = (item_input, action_value_bin, action_index)
        return item


def trim_list(lst, k):
    """
    keep at most k elements from list lst
    """
    return lst[-k:] if len(lst) > k else lst


def train(
        criterion: nn.NLLLoss,
        optimizer: AdamW,
        dataset: MyDataset, batch_size: int, shuffle_train_examples: bool, agent: Agent,
        max_grad_norm: float, device: torch.device,
):
    action_value_network = agent.action_value_network()
    action_value_network.train()

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_train_examples)

    data = next(iter(train_loader))

    optimizer.zero_grad()
    (inputs, targets, action_indices) = data

    inputs = [t.to(device) for t in inputs]
    targets = targets.to(device)

    # outputs has shape [batch_size, num_actions, num_classes].
    # We need a shape of [batch_size, num_classes] to use the criterion.
    all_predicted_action_values = action_value_network(inputs)

    outputs = all_predicted_action_values[torch.arange(
        batch_size), action_indices]

    loss = criterion(outputs, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        action_value_network.parameters(), max_grad_norm)
    optimizer.step()
    loss = loss.cpu().item()

    for t in inputs:
        del t
    del targets
    del outputs

    agent.step_policy_network(
        inputs, all_predicted_action_values.detach(), action_indices)

    return loss


def print_train_examples(train_action_examples):
    print("Train Examples")
    print(len(train_action_examples))
    for (idx, example) in enumerate(train_action_examples):
        print("---------------------------")
        print("Example: " + str(idx))
        example_input = example[0]
        example_target = example[1]

        print("example_input")
        print(tokens_to_text(example_input))
        print("example_target")
        print(example_target)


def print_model_outputs_for_train_examples(dataset: MyDataset, batch_size: int, agent: Agent, device: torch.device,):
    print("[after training] print_model_outputs_for_train_examples")
    action_value_network = agent.action_value_network()
    inference_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    for data in inference_loader:
        (inputs, targets, action_indices) = data
        inputs = [t.to(device) for t in inputs]
        targets = targets.to(device)
        outputs = action_value_network(inputs)
        outputs = outputs[torch.arange(batch_size), action_indices]
        for idx in range(len(inputs[0])):
            print("--------------------")
            print(f"idx: {idx} ")
            input = [inputs[0][idx], inputs[1][idx],
                     inputs[2][idx]]
            target = targets[idx].item()
            output = outputs[idx].argmax(dim=-1).item()
            print("Example: " + str(idx))
            print("input")
            print("".join(
                list(map(chr, input[0].tolist() + input[1].tolist() + input[2].tolist()))))
            print("target: ")
            print(target)
            print("output: ")
            print(output)
            # Only check first example from batch for now.
            break
        for t in inputs:
            del t
        del targets
        del outputs
        # Only check first batch for now.
        break


def get_grad_norm(model):
    """
    Calculates and prints the total L2 norm of the gradients of all model parameters.

    Args:
    model (torch.nn.Module): The PyTorch model whose gradients you want to monitor.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            # Square for L2 norm
            param_norm = p.grad.data.norm(2).item()**2
            total_norm += param_norm
        # Take the square root for L2 norm
        total_norm = total_norm**0.5

    return total_norm


def train_model_using_experience_replay(
    config: Configuration,
    context_size: int, batch_size: int, device: torch.device,
    agent: Agent, total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool, lr: float, weight_decay: float,
    max_grad_norm: float, print_model_outputs: bool, save_step_losses: bool,
    num_steps: int,
):
    criterion = nn.NLLLoss()
    optimizer = AdamW(agent.action_value_network().parameters(),
                      lr=lr, weight_decay=weight_decay)

    environment = Environment(cell_value_size)
    steps = []
    losses = []

    experience_replay_data_set = []
    for step in range(num_steps):
        if step % config.target_network_update_period == 0:
            agent.update_target_action_value_network()

        experience_replay_data_set, loss = train_model_with_experience_replay_data_set(
            config,
            environment,
            criterion,
            optimizer,
            experience_replay_data_set,
            context_size, batch_size, device,
            agent,
            total_train_examples,
            puzzle_train_examples, cell_value_size,
            discount, padding_char, num_classes, shuffle_train_examples,
            max_grad_norm, print_model_outputs,
        )

        if loss != None:
            steps.append(step)
            losses.append(loss)
            print(f"Step: {step}/{num_steps}  loss: {loss:.8f}")

    dynamic_time_marker = datetime.now(timezone.utc).isoformat()
    train_loss_csv_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.csv"
    train_loss_png_path = f"/workspace/reports/{dynamic_time_marker}-step_loss.png"

    if save_step_losses:
        df = pd.DataFrame(data={'step': steps, 'loss': losses})
        df.to_csv(train_loss_csv_path, index=False)
        plot_train_loss_graph(steps, losses, train_loss_png_path)

    # Plot total rewards per episode
    total_rewards_csv_path = f"/workspace/reports/{dynamic_time_marker}-total_rewards.csv"
    total_rewards_png_path = f"/workspace/reports/{dynamic_time_marker}-total_rewards.png"
    total_rewards = environment.get_total_rewards_per_episode()
    episodes = range(len(total_rewards))
    df = pd.DataFrame(data={'episodes': episodes,
                      'total_rewards': total_rewards})
    df.to_csv(total_rewards_csv_path, index=False)
    plot_total_rewards_graph(episodes,
                             total_rewards, total_rewards_png_path)


def train_model_with_experience_replay_data_set(
    config: Configuration,
    environment: Environment,
    criterion: nn.NLLLoss,
    optimizer: AdamW,
    experience_replay_data_set: List[Experience],
    context_size: int, batch_size: int, device: torch.device,
    agent: Agent,
    total_train_examples: int,
    puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    cell_value_size: int, discount: float, padding_char: str, num_classes: int,
    shuffle_train_examples: bool,
    max_grad_norm: float, print_model_outputs: bool,
) -> List[Experience]:
    """
    See:
    Human-level control through deep reinforcement learning
    https://www.nature.com/articles/nature14236
    """

    new_train_examples = play_game_using_model(
        environment,
        config.max_taken_actions_per_step,
        padding_char,
        context_size,
        batch_size,
        device,
        agent,
        puzzle_train_examples, cell_value_size)

    experience_replay_data_set_size = 4096
    experience_replay_data_set += new_train_examples
    experience_replay_data_set = trim_list(
        experience_replay_data_set,
        experience_replay_data_set_size,
    )

    min_experience_replay_data_set_size = 4 * batch_size
    loss = None

    if len(experience_replay_data_set) >= min_experience_replay_data_set_size:
        dataset = MyDataset(
            experience_replay_data_set, config, device, agent,)

        loss = train(
            criterion,
            optimizer,
            dataset, batch_size, shuffle_train_examples, agent,
            max_grad_norm, device,)

        if print_model_outputs:
            print_model_outputs_for_train_examples(
                dataset, batch_size, agent, device,)

    return experience_replay_data_set, loss
