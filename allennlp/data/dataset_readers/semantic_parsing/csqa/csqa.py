
"""
Reader for CSQA (https://amritasaha1812.github.io/CSQA/).
"""

from typing import Dict, List, Any
import gzip
import json
import logging
import os

from overrides import overrides
from collections import defaultdict

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import CSQAKnowledgeGraph, CSQAContext
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# noinspection PyTypeChecker
@DatasetReader.register("csqa")
class CSQADatasetReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 dpd_output_directory: str = None,
                 max_dpd_logical_forms: int = 10,
                 sort_dpd_logical_forms: bool = True,
                 max_dpd_tries: int = 20,
                 keep_if_no_dpd: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 kg_path: str = None,
                 entity_id2string_path: str = None,
                 predicate_id2string_path: str = None,
                 read_only_direct: bool = True
                 ) -> None:
        super().__init__(lazy=lazy)
        self._dpd_output_directory = dpd_output_directory
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._sort_dpd_logical_forms = sort_dpd_logical_forms
        self._max_dpd_tries = max_dpd_tries
        self._keep_if_no_dpd = keep_if_no_dpd
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.entity_id2string_path = entity_id2string_path
        self.kg_path = kg_path
        self.predicate_id2string_path = predicate_id2string_path
        self.load_direct_questions_only = read_only_direct

    @overrides
    def _read(self, qa_path: str):
        if qa_path.endswith('.json'):
            yield from self._read_unprocessed_file(qa_path)
        elif qa_path.endswith('.somethingelse'):
            yield from self._read_preprocessed_file(qa_path)  # TODO: implement
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {qa_path}")

    def _read_unprocessed_file(self, qa_path: str):
        # Create context to get kg_data, entity_id2string, and predicate_id2string; We create them only once as they
        # are the same fore every instance.
        context = CSQAContext.read_from_file(self.kg_path, self.entity_id2string_path, self.predicate_id2string_path,
                                             [], [])
        kg_data = context.kg_data
        entity_id2string = context.entity_id2string
        predicate_id2string = context.predicate_id2string

        with open(qa_path) as f:
            data = json.load(f)
            data = iter(data)
            skip_next = False
            for qa_dict_question in data:
                qa_dict_question = defaultdict(str, qa_dict_question)
                qa_dict_answer = defaultdict(str, next(data))
                question = qa_dict_question["utterance"]
                question_entities = qa_dict_question["entities_in_utterance"]
                question_predicates = qa_dict_question["relations"]
                question_description = qa_dict_question["description"]
                question_type = qa_dict_question["question-type"]
                answer = qa_dict_answer["utterance"]
                answer_description = qa_dict_answer["description"]
                entities_result = qa_dict_answer["all_entities"]

                # TODO: do we need extra checks here (e.g. 2 clarifications in a row)?
                if skip_next:
                    skip_next = False
                    continue

                if self.load_direct_questions_only:
                    # If this question requires clarification, we skip the next (as it will contain a reference)
                    if "Clarification" in answer_description:
                        skip_next = True
                    if "Indirect" in question_description or "indirectly" in question_description or\
                            "Incomplete" in question_description or "Coreferenced" in question_type:
                        # skip , to next qa pair
                        continue

                # TODO: implement reading dynamic programming denotations
                if self._dpd_output_directory:
                    dpd_output_filename = os.path.join(self._dpd_output_directory,
                                                       "gz_filename")
                    try:
                        dpd_file = gzip.open(dpd_output_filename)
                        if self._sort_dpd_logical_forms:
                            logical_forms = [dpd_line.strip().decode('utf-8') for dpd_line in dpd_file]
                            # We'll sort by the number of open parens in the logical form, which
                            # tells you how many nodes there are in the syntax tree.
                            logical_forms.sort(key=lambda x: x.count('('))
                            if self._max_dpd_tries:
                                logical_forms = logical_forms[:self._max_dpd_tries]
                        else:
                            logical_forms = []
                            for dpd_line in dpd_file:
                                logical_forms.append(dpd_line.strip().decode('utf-8'))
                                if self._max_dpd_tries and len(logical_forms) >= self._max_dpd_tries:
                                    break
                    except FileNotFoundError:
                        logger.debug(f'Missing DPD output for instance ; skipping...')
                        logical_forms = None
                        if not self._keep_if_no_dpd:
                            continue
                else:
                    logical_forms = None

                instance = self.text_to_instance(question=question,
                                                 question_entities=question_entities,
                                                 question_predicates=question_predicates,
                                                 answer=answer,
                                                 entities_result=entities_result,
                                                 kg_data=kg_data,
                                                 entity_id2string=entity_id2string,
                                                 predicate_id2string=predicate_id2string,
                                                 dpd_output=logical_forms)
                yield instance

    def text_to_instance(self,
                         question: str,
                         question_entities: List[str],
                         question_predicates: List[str],
                         answer: str,
                         entities_result: List[str],
                         kg_data: List[Dict[str, str]],
                         entity_id2string: Dict[str, str],
                         predicate_id2string: Dict[str, str],
                         dpd_output: List[str] = None,
                         tokenized_question: List[Token] = None) -> object:
        """
        Reads text inputs and makes an instance.
        Parameters
        ----------
        question : ``str``
            Input question
        question_entities : ``List[str]``
            Entities in the question
        question_predicates : ``List[str]``
            Relations in the question
        answer : ``List[str]``
            Answer text (can differ from the actual result, e.g. naming only three of 100 found entities)
        entities_result : ``List[str]``
            Entities that constitute the result of the query executed to formulate the answer
        kg_data : ``List[Dict[str, str]]``
            Knowledge graph
        entity_id2string : ``Dice[str, str]``
            Mapping from entity ids to there string values
        predicate_id2string : ``Dict[str, str]``
            Mapping from predicate ids to there string values
        dpd_output : List[str], optional
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        tokenized_question : ``List[Token]``, optional
            If you have already tokenized the question, you can pass that in here, so we don't
            duplicate that work.  You might, for example, do batch processing on the questions in
            the whole dataset, then pass the result in here.
        """

        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        answer_field = TextField(tokenized_question, self._question_token_indexers)
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question],
                                    "answer": answer}

        context = CSQAContext.read_from_file('', '', '', tokenized_question, question_entities, kg_data=kg_data,
                                             entity_id2string=entity_id2string, predicate_id2string=predicate_id2string)
        language = CSQALanguage(context)

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)

        # Add empty rule (remove when loop above is implemented).
        action_field = ListField(production_rule_fields)
        world_field = MetadataField(language)

        fields = {'question': question_field,
                  # 'answer': answer_field,
                  'world': world_field,
                  'actions': action_field,
                  'metadata': MetadataField(metadata)}

        # TODO: Assuming that possible actions are the same in all worlds. This is not true of course
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

        # TODO: implement this part
        if dpd_output:
            action_sequence_fields: List[Field] = []
            for logical_form in dpd_output:
                if not self._should_keep_logical_form(logical_form):
                    continue
                try:
                    expression = language.parse_logical_form(logical_form)
                except ParsingError as error:
                    continue
                except:
                    raise
                action_sequence = language.get_action_sequence(expression)
                try:
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except KeyError as error:
                    continue
                if len(action_sequence_fields) >= self._max_dpd_logical_forms:
                    break

            if not action_sequence_fields:
                # This is not great, but we're only doing it when we're passed logical form
                # supervision, so we're expecting labeled logical forms, but we can't actually
                # produce the logical forms.  We should skip this instance.  Note that this affects
                # _dev_ and _test_ instances, too, so your metrics could be over-estimates on the
                # full test data.
                return None
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        else:
            # TODO: remove, this is just a placholder
            action_sequence_fields: List[Field] = []
            # # print()
            # for i in range(14,20):
            #     prod_rule = list(action_map.keys())[i]
            #     lhs, rhs = prod_rule.split("->")
            #     # if rhs[1] is not "P":
            #         # print(lhs)
            #         # print(rhs)
            #         # print()
            # get: 11, start 12,
            index_fields: List[Field] = [IndexField(12, action_field), IndexField(11, action_field),
                                         IndexField(588, action_field)]
            action_sequence_fields.append(ListField(index_fields))
            fields['target_action_sequences'] = ListField(action_sequence_fields)


        return Instance(fields)
