from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from typing import Dict, List, Set


class CSQAKnowledgeGraph(KnowledgeGraph):

    def __init__(self,
                 entities: Set[str],
                 neighbors: Dict[str, List[str]],
                 entity_text: Dict[str, str]) -> None:

        super().__init__(entities, neighbors, entity_text)

    @classmethod
    def read_from_file(cls, filename: str) -> 'CSQAKnowledgeGraph':
        return cls(set(), {}, {})
