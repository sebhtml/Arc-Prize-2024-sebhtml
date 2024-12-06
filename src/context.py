import torch
import copy
from typing import List, Tuple
from vision import Cell
from vision import VACANT_CELL_VALUE,  MASKED_CELL_VALUE,  OUTSIDE_CELL_VALUE
from q_learning import QLearningAction

VACANT_CELL_CHAR = '_'
MASKED_CELL_CHAR = 'X'
OUTSIDE_CELL_CHAR = '.'


class ExampleInputTokens:
    def __init__(self,
                 attended_example_input: List[int],
                 attended_current_state: List[int],
                 attended_action: List[int]):
        self.__attended_example_input = attended_example_input
        self.__attended_current_state = attended_current_state
        self.__attended_action = attended_action

    def attended_example_input(self) -> List[int]:
        return self.__attended_example_input

    def attended_current_state(self) -> List[int]:
        return self.__attended_current_state

    def attended_action(self) -> List[int]:
        return self.__attended_action


def actions_act_on_same_cell(action_1: QLearningAction, action_2: QLearningAction) -> bool:
    return (action_1.row(), action_1.col()) == (action_2.row(), action_2.col())


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
        attended_example_input: List[List[Cell]],
        attended_current_state: List[List[Cell]],
        action: QLearningAction, padding_char: str) -> ExampleInputTokens:
    """
    Tokenize a example input for the Q-network Q(s, a).
    Note that:
    - s contains the state which is (example_input, current_state)
    - a contains the action which is (next_state)
    """

    attended_example_input_text = ""
    attended_example_input_text += "exampleInput" + "\n"
    attended_example_input_text += state_to_text(attended_example_input)

    attended_current_state_text = ""
    attended_current_state_text += "currentState" + "\n"
    attended_current_state_text += state_to_text(attended_current_state)

    action_text = ""
    action_text += "action" + "\n"
    action_text += action_to_text(attended_current_state, action)

    return ExampleInputTokens(
        text_to_tokens(attended_example_input_text),
        text_to_tokens(attended_current_state_text),
        text_to_tokens(action_text)
    )


def text_to_tokens(s: str) -> List[int]:
    return list(map(ord, list(s)))


def tokens_to_text(example_input_tokens: ExampleInputTokens) -> str:
    tokens: List[int] = example_input_tokens.attended_example_input() + \
        example_input_tokens.attended_current_state(
    ) + example_input_tokens.attended_action()
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


def action_to_text(state: List[List[Cell]], action: QLearningAction) -> str:
    output = ""
    for row in range(len(state)):
        for col in range(len(state[row])):
            value = None
            if row == action.row() and col == action.col():
                value = str(action.cell_value())
            else:
                value = VACANT_CELL_CHAR
            output += value
        output += "\n"
    return output


def make_example_tensor(example_input_tokens: ExampleInputTokens, context_size: int):
    example_input = filter_tokens(
        example_input_tokens.attended_example_input())
    current_state = filter_tokens(
        example_input_tokens.attended_current_state())
    candidate_action = filter_tokens(example_input_tokens.attended_action())

    input_tokens: List[int] = example_input + current_state + candidate_action
    if len(input_tokens) > context_size:
        raise Exception(
            f"text ({len(input_tokens)} tokens) is too large to fit in context ! Increase context_size ({context_size})")
    item_input = [torch.tensor(example_input),
                  torch.tensor(current_state),
                  torch.tensor(candidate_action)
                  ]
    return item_input


def filter_tokens(tokens):
    return list(filter(filter_token, tokens))


def filter_token(token: int) -> bool:
    """
    The ASCII codes of characters '0' to '9' and of character '_'
    are the only allowed tokens in the context.
    """
    legal_tokens = list(map(lambda x: ord(str(x)), range(10))) + \
        list(map(ord, [VACANT_CELL_CHAR, MASKED_CELL_CHAR, OUTSIDE_CELL_CHAR]))
    return token in legal_tokens
