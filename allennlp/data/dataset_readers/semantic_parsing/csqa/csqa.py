
"""
Reader for CSQA (https://amritasaha1812.github.io/CSQA/).
"""

from typing import Dict, List, Any
import gzip
import json
import logging
import os

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse.contexts import CSQAKnowledgeGraph
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
                 question_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._dpd_output_directory = dpd_output_directory
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._sort_dpd_logical_forms = sort_dpd_logical_forms
        self._max_dpd_tries = max_dpd_tries
        self._keep_if_no_dpd = keep_if_no_dpd
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith('.json'):
            yield from self._read_unprocessed_file(file_path)
        # TODO: implement
        elif file_path.endswith('.somethingelse'):
            yield from self._read_preprocessed_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_unprocessed_file(self, file_path: str):
        with open(file_path) as data_file:
            data = json.load(data_file)

            for conversation_turn_dict in data:
                speaker = conversation_turn_dict["speaker"]
                utterance = conversation_turn_dict["utterance"]
                if speaker == "USER":
                    question = utterance
                elif speaker == "SYSTEM":
                    answer = utterance
                else:
                    raise AssertionError("Unexpected value of speaker", speaker)

                # TODO: implement reading dynamic programming denotations
                if self._dpd_output_directory:
                    dpd_output_filename = os.path.join(self._dpd_output_directory,
                                                       "gz_filename")
                    try:
                        dpd_file = gzip.open(dpd_output_filename)
                        if self._sort_dpd_logical_forms:
                            # TODO: are these sempre forms or other forms?
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

                if speaker == "SYSTEM":
                    instance = self.text_to_instance(question=question,
                                                     answer=answer,
                                                     dpd_output=logical_forms)
                    yield instance

    def text_to_instance(self,
                         question: str,
                         answer: str,
                         dpd_output: List[str] = None,
                         tokenized_question: List[Token] = None) -> Instance:
        """
        Reads text inputs and makes an instance.
        Parameters
        ----------
        question : ``str``
            Input question
        answer : ``str``
            Input answer
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

        # TODO: implement CSQAKnowledgeGraph class
        wikidata_knowledge_graph = CSQAKnowledgeGraph.read_from_file("somefile")
        world = CSQALanguage(wikidata_knowledge_graph)
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_productions():
            # TODO: implement, iterates over empty list now
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_table_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)

        # Add empty rule (remove when loop above is implemented).
        field = ProductionRuleField("", False)
        production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        # TODO: split answer if contains multiple entities?
        # TODO: create world

        fields = {'question': question_field,
                  'answer': answer_field,
                  'world': world_field,
                  'actions': action_field,
                  'metadata': MetadataField(metadata)}

        # TODO: do we have multiple target actions?
        action_sequence_fields: List[Field] = []
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

        if dpd_output:
            pass
            # TODO: set
            # fields['target_action_sequences'] = ListField(action_sequence_fields)
            # fields['agenda'] = ListField(agenda_index_fields)

        return Instance(fields)
