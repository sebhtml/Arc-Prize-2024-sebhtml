import torch
import copy
from typing import List
from vision import Cell, CellAddress, do_visual_fixation, crop_field_of_view
from vision import VACANT_CELL_VALUE, OUTSIDE_CELL_VALUE
from q_learning import QLearningAction
from model import CLS_TOKEN

VACANT_CELL_CHAR = '_'
OUTSIDE_CELL_CHAR = '.'


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


def tokenize_example_input(
        attended_example_input: List[List[Cell]],
        padding_char: str) -> Context:
    """
    Tokenize a example input.
    """

    attended_example_input_text = ""
    attended_example_input_text += "attendedExampleInput" + "\n"
    attended_example_input_text += state_to_text(attended_example_input)

    return Context(
        text_to_tokens(attended_example_input_text),
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def tokens_to_text(example_input_tokens: Context) -> str:
    tokens: List[int] = example_input_tokens.attended_example_input()
    return "".join(map(chr, tokens))


def state_to_text(state: List[List[Cell]]) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if state[row][col].cell_value() == VACANT_CELL_VALUE:
                value = VACANT_CELL_CHAR
            elif state[row][col].cell_value() == OUTSIDE_CELL_VALUE:
                value = OUTSIDE_CELL_CHAR
            else:
                value = str(state[row][col].cell_value())
            output += value
        output += "\n"
    return output


def make_example_tensor(example_input_tokens: Context, context_size: int):
    attended_example_input = filter_tokens(
        example_input_tokens.attended_example_input())

    input_tokens: List[int] = attended_example_input

    if len(input_tokens) > context_size:
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


def prepare_context(example_input: List[List[Cell]], cell_address: CellAddress,
                    padding_char: str, visual_fixation_width: int, visual_fixation_height: int,) -> Context:

    attented_example_input = do_visual_fixation(
        example_input, cell_address, visual_fixation_height, visual_fixation_width,)

    input_tokens = tokenize_example_input(
        attented_example_input, padding_char)

    return input_tokens
