import json
import time

from typing import Dict, List, Optional, Tuple, Union, Set
import pickle

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.table_question_context import TableQuestionContext


NUMBER_CHARACTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'}
MONTH_NUMBERS = {
    'january': 1,
    'jan': 1,
    'february': 2,
    'feb': 2,
    'march': 3,
    'mar': 3,
    'april': 4,
    'apr': 4,
    'may': 5,
    'june': 6,
    'jun': 6,
    'july': 7,
    'jul': 7,
    'august': 8,
    'aug': 8,
    'september': 9,
    'sep': 9,
    'october': 10,
    'oct': 10,
    'november': 11,
    'nov': 11,
    'december': 12,
    'dec': 12,
}
ORDER_OF_MAGNITUDE_WORDS = {'hundred': 100, 'thousand': 1000, 'million': 1000000}
NUMBER_WORDS = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    **MONTH_NUMBERS,
}


def _decode(o):
    # Note the "unicode" part is only for python2
    if isinstance(o, str):
        try:
            return int(o)
        except ValueError:
            return o
    elif isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [_decode(v) for v in o]
    else:
        return o


class CSQAContext:
    """
    Context for the CSQADomainlanguage. this context contains the knowledge graph, questions and some mappings from
    entity/predicate id's to their corresponding string value.

    #################################################################################################################
    IMPORTANT: CSQAContext objects can get very large (as they contains the full kg, therefore when initialize multiple
    CSQAContext objects, we should let every object point to the same kg dict. We can do this by 1) initializing one
    object using CSQAContext.read_from_file(path1,path2,path3) 2) read the kg_dict from the initialized object 3)
    initialize new CSQAContext objects by calling read_from_file(kg_dict), passing the kg_dict from the first object
    #################################################################################################################
    """

    def __init__(self,
                 kg_data: Dict[str, Dict[str, str]],
                 kg_type_data: Dict[str, Dict[str, str]],
                 question_tokens: List[Token],
                 question_entities: List[str],
                 question_predicates: List[str],
                 type_list: List[str],
                 entity_id2string: Dict[str, str],
                 predicate_id2string: Dict[str, str],
                 use_integer_ids=False) -> None:
        self.kg_data = kg_data
        self.kg_type_data = kg_type_data
        self.question_tokens = question_tokens
        self.question_entities = question_entities
        self.question_predicates = question_predicates
        self.question_type_list = type_list
        self.entity_id2string = entity_id2string
        self.predicate_id2string = predicate_id2string
        self.use_integer_ids = use_integer_ids

    def get_knowledge_graph(self):
        pass

    def get_entities_from_question(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        extracted_numbers = TableQuestionContext._get_numbers_from_tokens(self.question_tokens)
        return self.question_entities, extracted_numbers

    @classmethod
    def read_from_file(cls,
                       kg_path: str,
                       kg_type_data_path: str,
                       entity_id2string_path: str,
                       predicate_id2string_path: str,
                       question_tokens: List[Token],
                       question_entities: List[str],
                       question_predicates: List[str],
                       type_list: List[str],
                       kg_data: Dict[int, Dict[int, int]] = None,
                       kg_type_data: Dict[int, Dict[int, int]] = None,
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
            loaded entitity vocab
        predicate_id2string: ``Dict[str,str]``
            loaded predicate vocab

        Returns
        -------
        CSQAContext

        """
        if not kg_data:
            if '.json' in kg_path:
                use_integer_ids = False
                with open(kg_path, 'r') as file_pointer:
                    kg_data = json.load(file_pointer, object_hook=_decode)

            elif'.p' in kg_path:
                use_integer_ids = True
                if 'sample' not in kg_path:
                    print("Loading wikidata graph")
                with open(kg_path, 'rb') as file_pointer:
                    kg_data = pickle.load(file_pointer)
            else:
                raise ValueError()

        else:
            # inspect first key
            use_integer_ids = isinstance(next(iter(kg_data)), int)

        if not kg_type_data:
            if'.p' in kg_type_data_path:
                use_integer_ids = True
                if 'sample' not in kg_type_data_path:
                    print("Loading wikidata type graph")
                with open(kg_type_data_path, 'rb') as file_pointer:
                    kg_type_data = pickle.load(file_pointer)
            else:
                raise ValueError()

        if not entity_id2string:
            with open(entity_id2string_path, 'r') as file_pointer:
                entity_id2string = json.load(file_pointer)
        if not predicate_id2string:
            with open(predicate_id2string_path, 'r') as file_pointer:
                predicate_id2string = json.load(file_pointer)
        return cls(kg_data, kg_type_data, question_tokens, question_entities, question_predicates, type_list,
                   entity_id2string, predicate_id2string, use_integer_ids)
