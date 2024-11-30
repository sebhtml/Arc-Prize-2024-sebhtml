import torch
import numpy as np
import copy
from typing import List, Tuple
from file_storage import ExampleInputTokens
from context import get_puzzle_starting_state, get_state_texts
from context import tokenize_example_input, tokens_to_text
from vision import VACANT_CELL_CHAR, MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR
from vision import VACANT_CELL_VALUE
from vision import do_visual_fixation
from q_learning import QLearningAction, Cell, ReplayBuffer, Experience, GameState
from q_learning import reward, sum_of_future_rewards
from model import DecoderOnlyTransformerModel


def filter_token(token: int) -> bool:
    """
    The ASCII codes of characters '0' to '9' and of character '_'
    are the only allowed tokens in the context.
    """
    legal_tokens = list(map(lambda x: ord(str(x)), range(10))) + \
        list(map(ord, [VACANT_CELL_CHAR, MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR]))
    return token in legal_tokens


def filter_tokens(tokens):
    return list(filter(filter_token, tokens))


def make_example_tensor(example_input_tokens: ExampleInputTokens, context_size: int):
    example_input = filter_tokens(example_input_tokens._input_state)
    current_state = filter_tokens(example_input_tokens._current_state)
    candidate_action = filter_tokens(example_input_tokens._action)

    input_tokens: List[int] = example_input + current_state + candidate_action
    if len(input_tokens) > context_size:
        raise Exception(
            f"text ({len(input_tokens)} tokens) is too large to fit in context ! Increase context_size ({context_size})")
    item_input = [torch.tensor(example_input),
                  torch.tensor(current_state),
                  torch.tensor(candidate_action)
                  ]
    return item_input


def infer_action_value(model, input_text, context_size, device):
    inputs = make_example_tensor(input_text, context_size).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = model(inputs)
    action_value = outputs[0].argmax(dim=-1).item()
    return action_value


def print_inferred_action_value(model, input_text):
    action_value = infer_action_value(model, input_text)
    print(f"action_value: {action_value}")


def apply_puzzle_action_value_policy(puzzle_examples, model,
                                     padding_char: str, cell_value_size: int,
                                     context_size: int, batch_size: int,
                                     device,):
    for example_input, example_target in puzzle_examples:
        print("example")
        example_input = get_puzzle_starting_state(
            example_input, "input_state")
        current_state = get_puzzle_starting_state(
            example_target, "current_state")
        output_state = solve_puzzle_example_auto_regressive(
            example_input, current_state, model,
            padding_char, cell_value_size, context_size, batch_size,
            device,)
        print("final output_state")
        print_current_state(example_input, output_state, padding_char)
        # TODO make the code work to print the example_target.
        # print("Expected output")
        # print_current_state(
        # example_input, example_target, padding_char)


def solve_puzzle_example_auto_regressive(example_input, current_state, model, padding_char: str, cell_value_size: int,
                                         context_size: int, batch_size: int,
                                         device):
    model.eval()
    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(example_input, current_state, padding_char)

    puzzle_width = len(current_state[0])
    puzzle_height = len(current_state)

    verbose = True

    # Each cell is allowed to change exactly once.
    for _ in range(puzzle_width * puzzle_height):
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)
        np.random.shuffle(candidate_actions)

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            model,
            verbose,
        )

        if best_action == None:
            print_current_state(example_input, current_state, padding_char)
            raise Exception("Failed to select action")

        next_state = copy.deepcopy(current_state)
        row = best_action.row()
        col = best_action.col()
        cell_value = best_action.cell_value()
        next_state[row][col].set_value(cell_value)

        current_state = next_state
        print(f"best_next_state with {best_action_value}")
        print("AUTO-REGRESSIVE wannabe AGI megabot current state")
        print_current_state(example_input, current_state, padding_char)
    return current_state


def print_current_state(input_state, current_state, padding_char):
    input_state_text, current_state_text = get_state_texts(
        input_state, current_state, padding_char)

    print(input_state_text)
    print(current_state_text)


def select_action_with_deep_q_network(
        example_input: List[List[Cell]],
        current_state: List[List[Cell]],
        candidate_actions: list[QLearningAction],
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        verbose: bool,
):
    best_action = None
    best_action_value = None

    batch_tokens = []
    batch_inputs = []
    batch_actions = []

    for candidate_action_index in range(len(candidate_actions)):
        candidate_action = candidate_actions[candidate_action_index]

        (attented_example_input, attented_current_state, attented_candidate_action,
         translation_x, translation_y) = do_visual_fixation(example_input, current_state, candidate_action)

        input_tokens = tokenize_example_input(
            attented_example_input, attented_current_state, attented_candidate_action, padding_char)

        inputs = list(map(lambda tensor: tensor.unsqueeze(0),
                          make_example_tensor(input_tokens, context_size)))

        batch_tokens.append(input_tokens)
        batch_inputs.append(inputs)
        batch_actions.append(candidate_action)

        if len(batch_inputs) == batch_size or candidate_action_index == len(candidate_actions) - 1:
            # batch_tensors contains:
            # [
            #   [ tensor1, tensor2, tensor3],
            #   [ tensor1, tensor2, tensor3],
            #   [ tensor1, tensor2, tensor3],
            # ]
            inputs = [
                torch.cat(
                    list(map(lambda inputs: inputs[0], batch_inputs)), dim=0),
                torch.cat(
                    list(map(lambda inputs: inputs[1], batch_inputs)), dim=0),
                torch.cat(
                    list(map(lambda inputs: inputs[2], batch_inputs)), dim=0),
            ]
            inputs = [t.to(device) for t in inputs]
            outputs = model(inputs)

            for batch_index in range(len(batch_tokens)):
                input_tokens = batch_tokens[batch_index]
                candidate_action = batch_actions[batch_index]
                if verbose:
                    print("input_text")
                    print(tokens_to_text(input_tokens))
                action_value = outputs[batch_index].argmax(dim=-1).item()
                row = candidate_action.row()
                col = candidate_action.col()
                cell_value = candidate_action.cell_value()

                if verbose:
                    print(
                        f"Testing action  row: {row}  col: {col}  cell_value: {cell_value} action_value: {action_value}")
                if best_action_value == None or action_value > best_action_value:
                    best_action = candidate_action
                    best_action_value = action_value
            for t in inputs:
                del t
            del outputs
            # Clear accumulated batch.
            batch_tokens = []
            batch_inputs = []
            batch_actions = []

    return best_action, best_action_value


def play_game_using_model(
    padding_char: str,
    context_size: int,
    batch_size: int,
    device: torch.device,
    model: DecoderOnlyTransformerModel,
        puzzle_example, cell_value_size) -> ReplayBuffer:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by a random player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.
    For the probability of selecting a correct color, the reasoning is that a cell can take 1 color out of 10 colors.
    So, we use a probability of 1 / 10 = 0.1

    For any cell, 10% of the random games will have a correct value for that cell.
    """

    print("play_game_using_model")
    model.eval()
    replay_buffer = ReplayBuffer()

    (raw_example_input, raw_example_output) = puzzle_example

    input_width = len(raw_example_input[0])
    input_height = len(raw_example_input)
    output_width = len(raw_example_output[0])
    output_height = len(raw_example_output)

    if (input_width, input_height) != (output_width, output_height):
        raise Exception(
            f"input and output have different sizes: {(input_width, input_height)} and {(output_width, output_height)}")

    example_input = raw_example_input
    example_output = raw_example_output

    example_input = get_puzzle_starting_state(
        example_input, "input_state")
    current_state = get_puzzle_starting_state(
        example_output, "current_state")

    puzzle_width = len(current_state[0])
    puzzle_height = len(current_state)

    verbose = False

    # Each cell is allowed to change exactly once.
    for _ in range(puzzle_width * puzzle_height):
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)
        np.random.shuffle(candidate_actions)

        best_action, best_action_value = select_action_with_deep_q_network(
            example_input,
            current_state,
            candidate_actions,
            padding_char,
            context_size,
            batch_size,
            device,
            model,
            verbose,
        )

        row = best_action.row()
        col = best_action.col()
        new_value = best_action.cell_value()
        candidate_action = best_action

        # An action assigns a correct color or an incorrect color to a cell.
        # The only thing that matters is that we shuffled the legal actions before selecting the action.

        next_state = copy.deepcopy(current_state)
        next_state[row][col].set_value(new_value)

        immediate_reward = reward(example_output, candidate_action)

        experience = Experience(
            GameState(example_input, current_state),
            candidate_action,
            immediate_reward,
            GameState(example_input, next_state),
        )
        replay_buffer.add_experience(experience)
        current_state = next_state

    return replay_buffer


# TODO this function should receive just one puzzle example


def generate_train_action_examples(
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        puzzle_examples, cell_value_size, discount: float, padding_char: str
) -> List[Tuple[ExampleInputTokens, float]]:
    """
    Generate (state, action, action_value) experience examples for all puzzle examples.
    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value)
    until the end of the game is reached.
    This is essentially a wrong incorrect half-baked MCTS (Monte Carlo tree search) in the sense that it's a
    tree search. And there is no Monte Carlo or Markov Chain involved here since we are lazy at best.
    """
    train_examples = []
    for puzzle_example in puzzle_examples:
        replay_buffer = play_game_using_model(
            padding_char,
            context_size,
            batch_size,
            device,
            model,
            puzzle_example, cell_value_size)
        action_examples = extract_action_examples(
            replay_buffer, discount, padding_char)
        train_examples += action_examples
    return train_examples


def extract_action_examples(replay_buffer: ReplayBuffer, discount: float, padding_char: str) -> List[Tuple[ExampleInputTokens, float]]:
    """
    This software used reinforcement learning.
    It uses Q-learning.

    See https://en.wikipedia.org/wiki/Q-learning
    See https://en.wikipedia.org/wiki/Bellman_equation

    See https://www.science.org/doi/10.1126/science.153.3731.34
    """

    examples = []

    experiences = replay_buffer.experiences()
    for experience in experiences:
        immediate_reward = experience.reward()
        example_input = experience.state().example_input()
        current_state = experience.state().current_state()
        candidate_action = experience.action()

        (attented_example_input, attented_current_state, attented_candidate_action,
         translation_x, translation_y) = do_visual_fixation(
            example_input, current_state, candidate_action)

        input_tokens = tokenize_example_input(
            attented_example_input, attented_current_state, attented_candidate_action, padding_char)

        expected_rewards = sum_of_future_rewards(
            immediate_reward, discount, attented_current_state, attented_candidate_action)
        action_value = expected_rewards
        example = (input_tokens, action_value)

        examples.append(example)

    return examples


def generate_examples(
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
    total_train_examples: int, puzzle_train_examples, cell_value_size: int,
        discount: float, padding_char: str
) -> List[Tuple[ExampleInputTokens, float]]:
    generated_examples = []
    must_generate_more_examples = True
    last_counter = 0
    step = 1
    while must_generate_more_examples:
        must_print = False
        examples = generate_train_action_examples(
            context_size,
            batch_size,
            device,
            model,
            puzzle_train_examples, cell_value_size, discount, padding_char)

        generated_examples += examples
        if len(generated_examples) >= last_counter + step:
            must_print = True
        if len(generated_examples) >= total_train_examples:
            must_generate_more_examples = False
            must_print = True
            break
        if must_print:
            print(
                f"Generating training examples... {len(generated_examples)}/{total_train_examples}")
            last_counter = len(generated_examples)
    return generated_examples


def generate_cell_actions(current_state, cell_value_size) -> list[QLearningAction]:
    candidate_actions = []
    assert current_state != None
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            # A cell can only be changed once.
            if current_state[row][col].value() != VACANT_CELL_VALUE:
                continue
            for new_value in range(cell_value_size):
                action = QLearningAction(row, col, new_value)
                candidate_actions.append(action)
    return candidate_actions
