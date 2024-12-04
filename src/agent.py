import torch
import numpy as np
import random
from typing import List, Tuple
from context import get_puzzle_starting_state, get_state_texts
from context import tokenize_example_input, tokens_to_text, make_example_tensor
from context import ExampleInputTokens
from vision import do_visual_fixation
from q_learning import QLearningAction, Cell, ReplayBuffer, Experience, GameState
from q_learning import sum_of_future_rewards
from model import DecoderOnlyTransformerModel
from emulator import Emulator


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
    emulator = Emulator(cell_value_size)
    for example_input, example_target in puzzle_examples:
        print("example")
        solve_puzzle_example_auto_regressive(
            emulator,
            example_input, model,
            padding_char, context_size, batch_size,
            device,)

        example_input, current_state = emulator.game_state()

        print("final output_state")
        print_current_state(example_input, current_state, padding_char)

        # TODO make the code work to print the example_target.
        # print("Expected output")
        # print_current_state(
        # example_input, example_target, padding_char)


def solve_puzzle_example_auto_regressive(emulator: Emulator,
                                         example_input: List[List[int]], model: DecoderOnlyTransformerModel, padding_char: str,
                                         context_size: int, batch_size: int,
                                         device: torch.device):
    model.eval()

    emulator.set_puzzle_example(example_input, None)

    example_input, current_state = emulator.game_state()

    print("AUTO-REGRESSIVE wannabe AGI megabot current state")
    print_current_state(example_input, current_state, padding_char)

    verbose = True

    while not emulator.is_in_terminal_state():
        candidate_actions = emulator.list_actions()

        example_input, current_state = emulator.game_state()

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

        immediate_reward = emulator.take_action(best_action)

        example_input, current_state = emulator.game_state()

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
    np.random.shuffle(candidate_actions)

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
        emulator: Emulator,
        padding_char: str,
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]], cell_value_size: int) -> ReplayBuffer:
    """
    Generate (state, action, reward, next_state) experiences from a simulated game of the puzzle by a random player.

    Each time that the player assigns a color to a cell, the assigned color is either correct or incorrect.

    We start from an empty board, and generate legal actions, and choose the best action (argmax of action value),
    until the end of the game is reached.
    """

    model.eval()
    replay_buffer = ReplayBuffer()

    if emulator.is_in_terminal_state():
        i = random.randrange(0, len(puzzle_train_examples))
        puzzle_example = puzzle_train_examples[i]

        (raw_example_input, raw_example_output) = puzzle_example
        emulator.set_puzzle_example(raw_example_input, raw_example_output)

    verbose = False

    while not emulator.is_in_terminal_state():
        candidate_actions = emulator.list_actions()

        example_input, current_state = emulator.game_state()

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

        immediate_reward = emulator.take_action(best_action)

        example_input, next_state = emulator.game_state()

        experience = Experience(
            GameState(example_input, current_state),
            best_action,
            immediate_reward,
            GameState(example_input, next_state),
        )
        replay_buffer.add_experience(experience)

    return replay_buffer


def extract_action_examples(replay_buffer: ReplayBuffer, discount: float, padding_char: str) -> List[Tuple[ExampleInputTokens, float]]:
    """
    Generate (state, action, action_value) action examples for a puzzle example.
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
        emulator: Emulator,
        context_size: int,
        batch_size: int,
        device: torch.device,
        model: DecoderOnlyTransformerModel,
        puzzle_train_examples: List[Tuple[List[List[int]], List[List[int]]]], cell_value_size: int,
        discount: float, padding_char: str
) -> List[Tuple[ExampleInputTokens, float]]:
    """
    Generate training examples from a puzzle example.
    """

    replay_buffer = play_game_using_model(
        emulator,
        padding_char,
        context_size,
        batch_size,
        device,
        model,
        puzzle_train_examples, cell_value_size)

    action_examples = extract_action_examples(
        replay_buffer, discount, padding_char)

    return action_examples
