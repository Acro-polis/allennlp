import json
from typing import Dict, List, Optional, Tuple, Union, Set

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

    def get_entities_from_question(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        extracted_numbers = TableQuestionContext._get_numbers_from_tokens(self.question_tokens)
        return self.question_entities, extracted_numbers


    @staticmethod
    def _get_numbers_from_tokens(tokens: List[Token]) -> List[Tuple[str, int]]:
        """
        Finds numbers in the input tokens and returns them as strings.  We do some simple heuristic
        number recognition, finding ordinals and cardinals expressed as text ("one", "first",
        etc.), as well as numerals ("7th", "3rd"), months (mapping "july" to 7), and units
        ("1ghz").

        We also handle year ranges expressed as decade or centuries ("1800s" or "1950s"), adding
        the endpoints of the range as possible numbers to generate.

        We return a list of tuples, where each tuple is the (number_string, token_index) for a
        number found in the input tokens.
        """
        numbers = []
        for i, token in enumerate(tokens):
            number: Union[int, float] = None
            token_text = token.text
            text = token.text.replace(',', '').lower()
            if text in NUMBER_WORDS:
                number = NUMBER_WORDS[text]

            magnitude = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ORDER_OF_MAGNITUDE_WORDS:
                    magnitude = ORDER_OF_MAGNITUDE_WORDS[next_token]
                    token_text += ' ' + tokens[i + 1].text

            is_range = False
            if len(text) > 1 and text[-1] == 's' and text[-2] == '0':
                is_range = True
                text = text[:-1]

            # We strip out any non-digit characters, to capture things like '7th', or '1ghz'.  The
            # way we're doing this could lead to false positives for something like '1e2', but
            # we'll take that risk.  It shouldn't be a big deal.
            text = ''.join(text[i] for i, char in enumerate(text) if char in NUMBER_CHARACTERS)

            try:
                # We'll use a check for float(text) to find numbers, because text.isdigit() doesn't
                # catch things like "-3" or "0.07".
                number = float(text)
            except ValueError:
                pass

            if number is not None:
                number = number * magnitude
                if '.' in text:
                    number_string = '%.3f' % number
                else:
                    number_string = '%d' % number
                numbers.append((number_string, i))
                if is_range:
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == '0':
                        num_zeros += 1
                    numbers.append((str(int(number + 10 ** num_zeros)), i))
        return numbers

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
                predicate_id2string = json.load(file_pointer)
        return cls(kg_data, question_tokens, question_entities, entity_id2string, predicate_id2string)
