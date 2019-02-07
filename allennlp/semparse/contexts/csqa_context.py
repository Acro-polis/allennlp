import json
from typing import List, Dict

from allennlp.data.tokenizers import Token


class CSQAContext:

    def __init__(self,
                 kg_data: List[Dict[str, str]],
                 question_tokens: List[Token]) -> None:

        self.kg_data = kg_data
        self.question_tokens = question_tokens

    def get_knowledge_graph(self):
        pass

    def get_entities_from_question(self):
        pass

    @classmethod
    def read_from_json(cls,
                       kg_dict: Dict[str, Dict[str, List[str]]],
                       question_tokens: List[Token]) -> "CSQAContext":

        # TODO: check: I believe we need a List as inner data structure
        # TODO: map entity/predicate ID's to strings
        kg_data: List[Dict[str, List[str]]] = []
        for subject in kg_dict.keys():
            predicate_object_dict = kg_dict[subject]
            kg_data.append(predicate_object_dict)
            # predicates = kg_dict[subject].keys()
            # for predicate in predicates:
            #     objects = kg_dict[subject][predicate]
            #     kg_data.append({predicate: objects})
        return cls(kg_data, question_tokens)

    @classmethod
    def read_from_file(cls, filename: str, question_tokens: List[Token]) -> 'CSQAContext':
        with open(filename, 'r') as file_pointer:
            kg_dict = json.load(file_pointer)
            return cls.read_from_json(kg_dict, question_tokens)

