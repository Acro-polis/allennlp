from typing import Set, Dict

from allennlp.semparse.worlds.world import World
from allennlp.semparse.contexts import CSQAKnowledgeGraph

from nltk.sem.logic import Type

from overrides import overrides


class CSQAWorld(World):

    def __init__(self, knowledge_graph: CSQAKnowledgeGraph) -> None:
        super(CSQAWorld, self).__init__(constant_type_prefixes=None,
                                        global_type_signatures=None,
                                        global_name_mapping=None,
                                        num_nested_lambdas=0)
        self.table_graph = knowledge_graph

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return []

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return []

    @overrides
    def _get_curried_functions(self) -> Dict[str, int]:
        pass

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        pass
