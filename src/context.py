import torch
import numpy as np
import copy
from typing import List
from vision import Cell, CellAddress
from vision import select_visual_fixations
from vision import VACANT_CELL_VALUE, OUTSIDE_CELL_VALUE
from vision import VACANT_CELL_CHAR, OUTSIDE_CELL_CHAR
from q_learning import ExampleInput
from model import CLS_TOKEN


class Context:
    def __init__(self,
                 attended_example_input: List[int],
                 ):
        self.__attended_example_input = attended_example_input

    def attended_example_input(self) -> List[int]:
        return self.__attended_example_input


def get_puzzle_starting_state(state: List[List[Cell]]) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = VACANT_CELL_VALUE
            current_state[row][col] = Cell(value)
    return current_state


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def tokens_to_text(example_input_tokens: Context) -> str:
    tokens: List[int] = example_input_tokens.attended_example_input()
    return "".join(map(chr, tokens))


def make_example_tensor(example_input_tokens: Context, context_size: int):
    attended_example_input = filter_tokens(
        example_input_tokens.attended_example_input())

    input_tokens: List[int] = attended_example_input

    if len(input_tokens) > context_size:
        print(f"context")
        print(tokens_to_text(example_input_tokens))
        raise Exception(
            f"text ({len(input_tokens)} tokens) is too large to fit in context ! Increase context_size ({context_size})")
    item_input = torch.tensor(attended_example_input)
    return item_input


def filter_tokens(tokens):
    return list(filter(filter_token, tokens))


def filter_token(token: int) -> bool:
    """
    The ASCII codes of characters '0' to '9' and of character '_'
    are the only allowed tokens in the context.
    """
    if token == CLS_TOKEN:
        raise Exception(
            "CLS token is forbidden before passing the tensor to the neural net")
    legal_tokens = list(map(lambda x: ord(str(x)), range(10))) + \
        list(map(ord, [VACANT_CELL_CHAR,
             OUTSIDE_CELL_CHAR]))
    return token in legal_tokens


def prepare_context(example_input_obj: ExampleInput, cell_address: CellAddress,
                    padding_char: str,
                    num_visual_fixations: int,
                    visual_fixation_height: int,
                    visual_fixation_width: int,
                    ) -> Context:

    # Get salient visual fixations.
    example_input = example_input_obj.cells()
    state = []
    for row in example_input:
        row2 = list(map(lambda cell: cell.cell_value(), row))
        state.append(row2)
    cell_addresses = select_visual_fixations(
        state, num_visual_fixations, visual_fixation_height, visual_fixation_width,)
    np.random.shuffle(cell_addresses)

    # Append the cell address to attend to.
    cell_addresses.append(cell_address)

    context_text = ""

    for cell_address in cell_addresses:
        context_text += "fixation" + "\n"
        context_text += example_input_obj.visual_fixation_text(
            cell_address, visual_fixation_height, visual_fixation_width,)

    context = Context(
        text_to_tokens(context_text),
    )

    return context
