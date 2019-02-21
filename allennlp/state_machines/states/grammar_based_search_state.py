from typing import Any, Dict, List, Sequence, Tuple, Generic, TypeVar

T = TypeVar('T', bound='GrammarBasedSearchState')
import torch
from copy import deepcopy

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.rnn_statelet import RnnStatelet
from allennlp.state_machines.states.grammar_statelet import ActionRepresentation
from allennlp.state_machines.states.state import State


class GrammarBasedSearchState:
    """

    Parameters
    ----------
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    grammar_state : ``List[GrammarStatelet]``
        This hold the current grammar state for each element of the group.  The ``GrammarStatelet``
        keeps track of which actions are currently valid.
    """
    def __init__(self,
                 action_history: List[str],
                 grammar_state: GrammarStatelet) -> None:
        self.action_history = action_history
        self.grammar_state = grammar_state

    def get_valid_actions(self) -> ActionRepresentation:
        """
        Returns a list of valid actions for each element of the group.
        """
        return self.grammar_state.get_valid_actions()

    def is_finished(self) -> bool:
        return self.grammar_state.is_finished()

    def take_action(self, action: str):
        new_action_history = deepcopy(self.action_history)
        new_action_history.append(action)
        return GrammarBasedSearchState(action_history=new_action_history,
                                       grammar_state=self.grammar_state.take_action(action))
