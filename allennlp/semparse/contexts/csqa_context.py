import json
from typing import List, Dict

from allennlp.data.tokenizers import Token


class CSQAContext:

    def __init__(self,
                 kg_data: List[Dict[str, str]],
                 question_tokens: List[Token],
                 question_entities: List[str],
                 entity_id2string: Dict[str, str],
                 predicate_id2string: Dict[str, str]) -> None:
        self.kg_data = kg_data
        self.question_tokens = question_tokens
        self.question_entities = question_entities
        self.entity_id2string = entity_id2string
        self.predicate_id2string = predicate_id2string

    def get_knowledge_graph(self):
        pass

    def get_entities_from_question(self):
        pass

    @classmethod
    def read_kg_from_json(cls,
                       kg_dict: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
        # TODO: check: I believe we need a List as inner datastrucure
        kg_data: List[Dict[str, List[str]]] = []
        for subject in kg_dict.keys():
            predicate_object_dict = kg_dict[subject]
            kg_data.append(predicate_object_dict)
        return kg_data

    @classmethod
    def read_from_file(cls,
                       kg_path: str,
                       entity_id2string_path: str,
                       predicate_id2string_path: str,
                       question_tokens: List[Token],
                       question_entities: List[str],
                       kg_data: List[Dict[str, str]] = None,
                       entity_id2string: Dict[str, str] = None,
                       predicate_id2string: Dict[str, str] = None
                       ) -> 'CSQAContext':
        if not kg_data:
            with open(kg_path, 'r') as file_pointer:
                kg_dict = json.load(file_pointer)
                kg_data = cls.read_kg_from_json(kg_dict)
        if not entity_id2string:
            with open(entity_id2string_path, 'r') as file_pointer:
                entity_id2string = json.load(file_pointer)
        if not predicate_id2string:
            with open(predicate_id2string_path, 'r') as file_pointer:
                entity_id2string = json.load(file_pointer)
        return cls(kg_data, question_tokens, question_entities, entity_id2string, predicate_id2string)
