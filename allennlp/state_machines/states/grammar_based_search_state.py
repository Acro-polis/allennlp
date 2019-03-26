from typing import Any, Dict, List, Sequence, Tuple, Generic, TypeVar

from allennlp.nn import util
from allennlp.state_machines.states import GrammarStatelet

T = TypeVar('T', bound='GrammarBasedSearchState')
import torch
from copy import deepcopy
from typing import Callable, Dict, Generic, List, TypeVar

from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.grammar_statelet import ActionRepresentation


class GrammarBasedSearchState:
    """

    Parameters
    ----------
    action_history : ``List[str]``
        Passed to super class; see docs there.
    grammar_state : ``GrammarStatelet``
        This hold the current grammar state for each element of the group.  The ``GrammarStatelet``
        keeps track of which actions are currently valid.
    """

    def __init__(self,
                 action_history: List[str],
                 # grammar_state: GrammarStatelet,
                 nonterminal_stack: List[str],
                 valid_actions: Dict[str, ActionRepresentation],
                 is_nonterminal: Callable[[str], bool],
                 reverse_productions: bool = True) -> None:
        self.action_history = action_history
        # self.grammar_state = grammar_state


        self._nonterminal_stack = nonterminal_stack
        self._valid_actions = valid_actions
        self._is_nonterminal = is_nonterminal
        self._reverse_productions = reverse_productions

    def is_finished(self) -> bool:
        return not self._nonterminal_stack

    def get_valid_actions(self) -> ActionRepresentation:
        return self._valid_actions[self._nonterminal_stack[-1]]

    def take_action(self, action: str):
        new_action_history = deepcopy(self.action_history)
        new_action_history.append(action)
        left_side, right_side = action.split(' -> ')
        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          f"but got rule {left_side} -> {right_side}")

        new_stack = self._nonterminal_stack[:-1]

        if right_side[0] == '[':
            productions = right_side[1:-1].split(', ')
        else:
            productions = [right_side]

        if self._reverse_productions:
            productions = list(reversed(productions))

        for production in productions:
            if self._is_nonterminal(production):
                new_stack.append(production)

        return GrammarBasedSearchState(action_history=new_action_history,
                                       nonterminal_stack=new_stack,
                                       valid_actions=self._valid_actions,
                                       is_nonterminal=self._is_nonterminal,
                                       reverse_productions=self._reverse_productions)

    @staticmethod
    def _get_productions_from_string(self, production_string: str) -> List[str]:
        """
        Takes a string like '[<d,d>, d]' and parses it into a list like ['<d,d>', 'd'].  For
        production strings that are not lists, like '<e,d>', we return a single-element list:
        ['<e,d>'].
        """
        if production_string[0] == '[':
            return production_string[1:-1].split(', ')
        else:
            return [production_string]

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            # pylint: disable=protected-access
            return all([
                self._nonterminal_stack == other._nonterminal_stack,
                util.tensors_equal(self._valid_actions, other._valid_actions),
                self._is_nonterminal == other._is_nonterminal,
                self._reverse_productions == other._reverse_productions,
                ])
        return NotImplemented
