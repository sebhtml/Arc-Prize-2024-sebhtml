import torch
import numpy as np
import copy
from file_storage import SampleInputTokens
from typing import List
from playout_simulation import get_puzzle_starting_state, get_state_texts, generate_cell_actions, do_visual_fixation
from playout_simulation import tokenize_sample_input, tokens_to_text
from vision import VACANT_CELL_CHAR, MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR


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


def make_sample_tensor(sample_input_tokens: SampleInputTokens, context_size: int):
    example_input = filter_tokens(sample_input_tokens._input_state)
    current_state = filter_tokens(sample_input_tokens._current_state)
    candidate_action = filter_tokens(sample_input_tokens._action)

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
    inputs = make_sample_tensor(input_text, context_size).unsqueeze(0)
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

    # Each cell is allowed to change exactly once.
    for _ in range(puzzle_width * puzzle_height):
        best_next_state = None
        best_action_value = None
        candidate_actions = generate_cell_actions(
            current_state, cell_value_size)
        np.random.shuffle(candidate_actions)

        batch_tokens = []
        batch_inputs = []
        batch_actions = []

        for candidate_action_index in range(len(candidate_actions)):
            candidate_action = candidate_actions[candidate_action_index]

            (attented_example_input, attented_current_state, attented_candidate_action, translation_x,
             translation_y) = do_visual_fixation(example_input, current_state, candidate_action)

            input_tokens = tokenize_sample_input(
                attented_example_input, attented_current_state, attented_candidate_action, padding_char)

            inputs = list(map(lambda tensor: tensor.unsqueeze(0),
                          make_sample_tensor(input_tokens, context_size)))

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
                    print("input_text")
                    print(tokens_to_text(input_tokens))
                    action_value = outputs[batch_index].argmax(dim=-1).item()
                    row = candidate_action.row()
                    col = candidate_action.col()
                    cell_value = candidate_action.cell_value()
                    next_state = copy.deepcopy(current_state)
                    next_state[row][col].set_value(cell_value)

                    print(
                        f"Testing action  row: {row}  col: {col}  cell_value: {cell_value} action_value: {action_value}")
                    if best_action_value == None or action_value > best_action_value:
                        best_next_state = next_state
                        best_action_value = action_value
                for t in inputs:
                    del t
                del outputs
                # Clear accumulated batch.
                batch_tokens = []
                batch_inputs = []
                batch_actions = []
        if best_next_state == None:
            print_current_state(example_input, current_state, padding_char)
            raise Exception("Failed to select action")
        current_state = best_next_state
        print(f"best_next_state with {best_action_value}")
        print("AUTO-REGRESSIVE wannabe AGI megabot current state")
        print_current_state(example_input, current_state, padding_char)
    return current_state


def print_current_state(input_state, current_state, padding_char):
    input_state_text, current_state_text = get_state_texts(
        input_state, current_state, padding_char)

    print(input_state_text)
    print(current_state_text)
