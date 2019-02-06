import json
from typing import List, Dict

from allennlp.data.tokenizers import Token


class CSQAContext:

    def __init__(self) -> None:
        self.kg_data = None

    def get_knowledge_graph(self):
        pass

    def get_entities_from_question(self):
        pass

    @classmethod
    def read_from_json(cls,
                       kg_dict: Dict[str, Dict[str, list[str]]],
                       question_tokens: List[Token]) -> 'CSQAContext':

        for entity in kg_dict.keys():
            predicate = js
        print(json)
        return None
        # return cls(table_data_with_column_types, column_types, question_tokens)

    @classmethod
    def read_from_file(cls, filename: str, question_tokens: List[Token]) -> 'CSQAContext':
        with open(filename, 'r') as file_pointer:
            kg_dict = json.load(file_pointer)
            return cls.read_from_json(kg_dict, question_tokens)

