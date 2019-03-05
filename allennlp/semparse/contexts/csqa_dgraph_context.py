from typing import Dict, List, Tuple
import json
import pydgraph

from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.table_question_context import TableQuestionContext


class CSQADgraphContext:
    """
    Context for the CSQADgraphDomainLanguage. this context contains the knowledge graph, questions and some mappings
    from entity/predicate id's to their corresponding string value.

    #################################################################################################################
    #################################################################################################################
    """

    def __init__(self,
                 kg_data: pydgraph.DgraphClient,
                 question_tokens: List[Token],
                 question_entities: List[str],
                 entity_id2string: Dict[str, str],
                 predicate_id2string: Dict[str, str]) -> None:
        self.kg_data = kg_data
        self.question_tokens = question_tokens
        self.question_entities = question_entities
        self.entity_id2string = entity_id2string
        self.predicate_id2string = predicate_id2string
        self.use_integer_id = False

    def get_entities_from_question(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        extracted_numbers = TableQuestionContext._get_numbers_from_tokens(self.question_tokens)
        return self.question_entities, extracted_numbers

    @classmethod
    def connect_to_db(cls,
                      ip: str,
                      entity_id2string_path: str,
                      predicate_id2string_path: str,
                      question_tokens: List[Token],
                      question_entities: List[str],
                      kg_data: pydgraph.DgraphClient = None,
                      entity_id2string: Dict[str, str] = None,
                      predicate_id2string: Dict[str, str] = None
                      ) -> "CSQADgraphContext":
        """
        This method loads a CSQAContext from file given the question tokens + the path to the kg,
        and entity and predicate dicts. Optionally, we can pass loaded dictionaries of each of those,
        which means the paths are ignored.
        Parameters
        ----------
        ip: ``str``, optional
            Ip address of the dgraph client stub
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
        if kg_data is None:
            client_stub = pydgraph.DgraphClientStub(ip)
            kg_data = pydgraph.DgraphClient(client_stub)
        if not entity_id2string:
            with open(entity_id2string_path, 'r') as file_pointer:
                entity_id2string = json.load(file_pointer)
        if not predicate_id2string:
            with open(predicate_id2string_path, 'r') as file_pointer:
                predicate_id2string = json.load(file_pointer)
        return cls(kg_data, question_tokens, question_entities, entity_id2string, predicate_id2string)
