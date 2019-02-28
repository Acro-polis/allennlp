import json
from typing import Dict, List, Tuple

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.table_question_context import TableQuestionContext


class CSQAContext:
    """
    Context for the CSQADomainLanguage. this context contains the knowledge graph, questions and some mappings from
    entity/predicate id's to their corresponding string value.

    #################################################################################################################
    IMPORTANT: CSQAContext objects can get very large (as they contains the full kg, therefore when initialize multiple
    CSQAContext objects, we should let every object point to the same kg dict. We can do this by 1) initializing one
    object using CSQAContext.read_from_file(path1,path2,path3) 2) read the kg_dict from the initialized object 3)
    initialize new CSQAContext objects by calling read_from_file(kg_dict), passing the kg_dict from the first object
    #################################################################################################################
    """

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

    def get_entities_from_question(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        extracted_numbers = TableQuestionContext._get_numbers_from_tokens(self.question_tokens)
        return self.question_entities, extracted_numbers

    @classmethod
    def read_kg_from_json(cls,
                          kg_dict: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
        # TODO: check: I believe we need a List as inner data structure
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
        """
        This method loads a CSQAContext from file given the question tokens + the path to the kg,
        and entity and predicate dicts. Optionally, we can pass loaded dictionaries of each of those,
        which means the paths are ignored.
        Parameters
        ----------
        kg_path: ``str``, optional
            Path to the knowledge graph file. We use this file to initialize our context
        entity_id2string_path: ``str``, optional
            Path to the json file which maps entity id's to their string values
        predicate_id2string_path: ``str``, optional
            Path to the json file which maps predicate id's to their string values
        question_tokens: ``List[Token]``
            question tokens
        question_entities: ``List[str]``
            list of entities
        kg_data: ``List[Dict[str,str]]``
            loaded knowledge graph
        entity_id2string: ``Dict[str,str]``
            loaded entity vocab
        predicate_id2string: ``Dict[str,str]``
            loaded predicate vocab

        Returns
        -------
        CSQAContext

        """
        if not kg_data:
            with open(kg_path, 'r') as file_pointer:
                kg_data = json.load(file_pointer)
                # kg_data = cls.read_kg_from_json(kg_dict)
        if not entity_id2string:
            with open(entity_id2string_path, 'r') as file_pointer:
                entity_id2string = json.load(file_pointer)
        if not predicate_id2string:
            with open(predicate_id2string_path, 'r') as file_pointer:
                predicate_id2string = json.load(file_pointer)
        return cls(kg_data, question_tokens, question_entities, entity_id2string, predicate_id2string)
