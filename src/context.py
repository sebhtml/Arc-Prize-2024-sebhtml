import torch
import copy
from typing import List
from vision import Cell, do_visual_fixation
from vision import VACANT_CELL_VALUE,  MASKED_CELL_VALUE,  OUTSIDE_CELL_VALUE
from q_learning import QLearningAction
from model import CLS_TOKEN

VACANT_CELL_CHAR = '_'
MASKED_CELL_CHAR = 'X'
OUTSIDE_CELL_CHAR = '.'


class Context:
    def __init__(self,
                 example_input: List[int],
                 current_state: List[int],
                 attended_example_input: List[int],
                 attended_current_state: List[int],
                 ):
        self.__example_input = example_input
        self.__current_state = current_state
        self.__attended_example_input = attended_example_input
        self.__attended_current_state = attended_current_state

    def example_input(self) -> List[int]:
        return self.__example_input

    def current_state(self) -> List[int]:
        return self.__current_state

    def attended_example_input(self) -> List[int]:
        return self.__attended_example_input

    def attended_current_state(self) -> List[int]:
        return self.__attended_current_state


def get_puzzle_starting_state(state, mode: str) -> List[List[Cell]]:
    current_state = copy.deepcopy(state)
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            value = VACANT_CELL_VALUE
            if mode == "example_input":
                value = state[row][col]
            current_state[row][col] = Cell(value)
    return current_state


def tokenize_example_input(
        example_input: List[List[Cell]],
        current_state: List[List[Cell]],
        attended_example_input: List[List[Cell]],
        attended_current_state: List[List[Cell]],
        padding_char: str) -> Context:
    """
    Tokenize a example input.
    """

    example_input_text = ""
    example_input_text += "exampleInput" + "\n"
    example_input_text += state_to_text(example_input)

    current_state_text = ""
    current_state_text += "currentState" + "\n"
    current_state_text += state_to_text(current_state)

    attended_example_input_text = ""
    attended_example_input_text += "attendedExampleInput" + "\n"
    attended_example_input_text += state_to_text(attended_example_input)

    attended_current_state_text = ""
    attended_current_state_text += "attendedCurrentState" + "\n"
    attended_current_state_text += state_to_text(attended_current_state)

    return Context(
        text_to_tokens(example_input_text),
        text_to_tokens(current_state_text),
        text_to_tokens(attended_example_input_text),
        text_to_tokens(attended_current_state_text),
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
            if state[row][col].value() == VACANT_CELL_VALUE:
                value = VACANT_CELL_CHAR
            elif state[row][col].value() == MASKED_CELL_VALUE:
                value = MASKED_CELL_CHAR
            elif state[row][col].value() == OUTSIDE_CELL_VALUE:
                value = OUTSIDE_CELL_CHAR
            else:
                value = str(state[row][col].value())
            output += value
        output += "\n"
    return output


def make_example_tensor(example_input_tokens: Context, context_size: int):
    example_input = filter_tokens(
        example_input_tokens.example_input())
    current_state = filter_tokens(
        example_input_tokens.current_state())
    attended_example_input = filter_tokens(
        example_input_tokens.attended_example_input())
    attended_current_state = filter_tokens(
        example_input_tokens.attended_current_state())

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
             MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR]))
    return token in legal_tokens


def prepare_context(example_input: List[List[Cell]], current_state: List[List[Cell]], candidate_action: QLearningAction,
                    padding_char: str) -> Context:

    masked_current_state = mask_current_state(current_state)

    (attented_example_input, attented_current_state,
     ) = do_visual_fixation(example_input, masked_current_state, candidate_action)

    input_tokens = tokenize_example_input(
        example_input, masked_current_state,
        attented_example_input, attented_current_state, padding_char)

    return input_tokens


def mask_current_state(current_state: List[List[Cell]]) -> List[List[Cell]]:
    """
    Mask non-vacant cells that were assigned by past actions to reduce combinatorics
    """
    masked_current_state = copy.deepcopy(current_state)
    width = len(masked_current_state[0])
    height = len(masked_current_state)
    for x in range(width):
        for y in range(height):
            if masked_current_state[y][x].value() != VACANT_CELL_VALUE:
                masked_current_state[y][x].set_value(MASKED_CELL_VALUE)
    return masked_current_state
