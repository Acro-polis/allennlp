"""
Reader for CSQA: https://amritasaha1812.github.io/CSQA/.
"""
import json
import logging
import os
import pathlib
import pickle
import tarfile
import numpy as np

from overrides import overrides
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, LabelField, ListField, ArrayField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, BertBasicWordSplitter
from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage
from allennlp.data.dataset_readers.semantic_parsing.csqa.util import get_dummy_action_sequences, question_is_indirect, \
    parse_answer, augment_with_context, get_extraction_dir, get_segment_field_from_tokens, prepare_question_for_bert
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

VERIFICATION_QUESTION_STRING = "Verification (Boolean) (All)"
RETRIEVAL_QUESTION_TYPES_DIRECT = ["Simple Question (Direct)", "Logical Reasoning (All)",
                                   "Quantitative Reasoning (All)", "Comparative Reasoning (All)"]
RETRIEVAL_QUESTION_TYPES_INDIRECT = ["Simple Question (Coreferenced)", "Simple Question (Ellipsis)", "Clarification"]
COUNT_QUESTION_TYPES = ["Quantitative Reasoning (Count) (All)", "Comparative Reasoning (Count) (All)"]
OTHER_QUESTION_TYPES = [VERIFICATION_QUESTION_STRING]

PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / ".." / ".." / "..").resolve()
FIXTURES_ROOT = PROJECT_ROOT / "allennlp" / "tests" / "fixtures"

# noinspection PyTypeChecker
@DatasetReader.register("csqa")
class CSQADatasetReader(DatasetReader):
    """
    This dataset reader is used for the CSQA dataset, which uses various files to contain different
    forms of information. QA files are parsed for question's and answer's type, text, entities and
    entity types. A knowledge graph (kg) is used to store entities and their relations. Another kg
    is used to store the types of the entities. Logical forms, i.e. queries that would result in
    the correct answers, can be specified for strong supervision training.

    To increase speed, pickles can be saved and loaded using this reader as well.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    dpd_output_file : ``str``, optional
        Directory that contains all the gzipped dpd output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``). This is required for training a model, but not required
        for prediction.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use for the questions. Will default to ``WordTokenizer()`` with Spacy's tagger
        enabled, as we use lemma matches as features for entity linking.
    question_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    kg_path : ``str``, optional
        Path to the knowledge graph file. We use this file to initialize our context.
    kg_type_path : ``str``, optional
        Path to the knowledge graph type file. We use this file to initialize our context.
    entity_id2string_path : ``str``, optional
        Path to the json file that maps entity id's to their string values.
    predicate_id2string_path : ``str``, optional
        Path to the json file that maps predicate id's to their string values.
    skip_approximate_questions : ``bool``, optional
        Boolean indicating whether questions containing "approximate" or "around" are skipped.
    read_only_direct : ``bool``, optional
        Boolean indicating whether only direct questions are read (without references to questions
        earlier in the conversation).
    """
    def __init__(self,
                 lazy: bool = False,
                 kg_path: str = None,
                 kg_type_path: str = None,
                 use_sample_kg: bool = False,
                 entity_id2string_path: str = None,
                 predicate_id2string_path: str = None,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 read_only_direct: bool = True,
                 skip_approximate_questions: bool = True,
                 augment_sentence_with_context: bool = False,
                 dpd_output_file: str = None
                 ) -> None:
        super().__init__(lazy=lazy)
        if not use_sample_kg:
            self.kg_path = cached_path(kg_path)
            self.kg_type_path = cached_path(kg_type_path)
        else:
            self.kg_path = cached_path(f'{FIXTURES_ROOT}/data/csqa/sample_kg.p')
            self.kg_type_path = cached_path(f'{FIXTURES_ROOT}/data/csqa/sample_kg.p')

        self.predicate_id2string_path = cached_path(predicate_id2string_path)
        self.entity_id2string_path = cached_path(entity_id2string_path)
        self.tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self.question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.load_direct_questions_only = read_only_direct
        self.skip_approximate_questions = skip_approximate_questions
        self.augment_sentence_with_context = augment_sentence_with_context
        self.dpd_output_file = dpd_output_file
        self.dpd_logical_form_dict = None
        self.use_bert_encoder = any(isinstance(idxr, PretrainedBertIndexer) for
                                    idxr in question_token_indexers.values())

    def init_dpd_dict(self):
        if not self.dpd_logical_form_dict and self.dpd_output_file:
            path = cached_path(self.dpd_output_file)
            with open(path, 'rb') as input_file:
                self.dpd_logical_form_dict = pickle.load(input_file)

    @overrides
    def _read(self, path: str):
        path = Path(cached_path(path))
        if path.match('*.json'):
            file_id = 'sample.json'
            yield from self._read_unprocessed_file(path, file_id)
        elif os.path.isdir(path):
            yield from self._read_from_directory(path)
        elif tarfile.is_tarfile(path):
            untarred_dir = get_extraction_dir(path)
            # check whether directory already exists (meaning the file has been extracted earlier)
            if os.path.isdir(untarred_dir):
                yield from self._read_from_directory(untarred_dir)
            else:
                print("Extracting into directory: ", untarred_dir)
                os.mkdir(untarred_dir)
                with tarfile.open(path) as tar:
                    for member in Tqdm.tqdm(tar.getmembers()):
                        tar.extract(member, path=untarred_dir)
                yield from self._read_from_directory(untarred_dir)
        else:
            raise ConfigurationError(f"Don't know how to read file type of {path}")

    def _read_from_directory(self, directory):
        directory = Path(directory)
        for file_path in directory.glob('**/*.json'):
            # The file_id is of format: QA_*/QA_*.json.
            file_id = file_path.relative_to(directory)
            yield from self._read_unprocessed_file(file_path, str(file_id))

    def _read_unprocessed_file(self, qa_file_path: str, file_id: str):
        self.init_dpd_dict()
        with open(qa_file_path) as f:
            data = iter(json.load(f))
            skip_next_turn = False
            turn_id = 0
            for qa_dict_question in data:
                qa_id = file_id + str(turn_id)
                turn_id += 1
                qa_dict_question = defaultdict(list, qa_dict_question)
                qa_dict_answer = defaultdict(list, next(data))
                instance, skip_next_turn = self._read_turn(qa_id,
                                                           qa_dict_question,
                                                           qa_dict_answer,
                                                           skip_next_turn)
                if instance:
                    yield instance

    def _read_turn(self, qa_id, qa_dict_question, qa_dict_answer, skip_turn):
        question = qa_dict_question["utterance"]
        question_description = qa_dict_question["description"]
        question_type_entities = qa_dict_question["type_list"]
        question_type = qa_dict_question["question-type"]
        answer_description = qa_dict_answer["description"]
        question_entities = qa_dict_question["entities_in_utterance"] + qa_dict_question["entities"]
        skip_next_turn = False

        if skip_turn:
            return None, skip_next_turn

        if self.load_direct_questions_only:
            # If this question requires clarification, the next question will contain a reference.
            skip_next_turn = "Clarification" in answer_description
            if question_is_indirect(question_description, question_type):
                return None, skip_next_turn

        if self.skip_approximate_questions and ("approximate" in question or "around" in question):
            return None, skip_next_turn

        if self.dpd_output_file:
            # TODO: fix difference between empty list and no key
            if qa_id in self.dpd_logical_form_dict:
                qa_logical_forms = self.dpd_logical_form_dict[qa_id]
                if not qa_logical_forms:
                    return None, skip_next_turn
            else:
                return None, skip_next_turn
        else:
            qa_logical_forms = get_dummy_action_sequences(question_entities, question_type_entities)

        instance = self.text_to_instance(qa_id=qa_id,
                                         question_type=question_type,
                                         question_description=question_description,
                                         question=question,
                                         question_entities=question_entities,
                                         question_predicates=qa_dict_question["relations"],
                                         answer=qa_dict_answer["utterance"],
                                         entities_result=qa_dict_answer["all_entities"],
                                         question_type_entities=question_type_entities,
                                         qa_logical_forms=qa_logical_forms)
        return instance, skip_next_turn

    def text_to_instance(self,
                         qa_id: str,
                         question_type: str,
                         question_description: str,
                         question: str,
                         question_entities: List[str],
                         question_type_entities: List[str],
                         question_predicates: List[str],
                         answer: str,
                         entities_result: List[str],
                         qa_logical_forms: List[str] = None) -> object:
        """
        Reads text inputs and creates an instance per question answer pair.

        Parameters
        ----------
        qa_id : ``str``
            Question answer turn id, e.g. 'train/QA_0/QA_1.json/0'.
        question_type : ``str``
            Type of the question, used to determine metrics per type of question.
        question_description : ``str``
            Description of the question in terms of 'Count', 'Indirect', etc.
        question : ``str``
            Input question.
        question_entities : ``List[str]``
            Entities in the question.
        question_type_entities : ``List[str]``
            Types of entities in the question.
        question_predicates : ``List[str]``
            Relations in the question.
        answer : ``str``
            Answer text (can differ from the actual result, e.g.: naming only three of 100 retrieved
            entities).
        entities_result : ``List[str]``
            Entities that constitute the result of the query executed to formulate the answer.
        kg_data : ``List[Dict[str, str]]``
            Knowledge graph.
        kg_type_data : ``List[Dict[str, str]]``
            Knowledge graph containing entity types.
        entity_id2string : ``Dice[str, str]``
            Mapping from entity ids to there string values.
        predicate_id2string : ``Dict[str, str]``
            Mapping from predicate ids to there string values.
        qa_logical_forms : List[str], optional
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        tokenized_question : ``List[Token]``, optional
            If you have already tokenized the question, you can pass that in here, so we don't
            duplicate that work. You might, for example, do batch processing on the questions in
            the whole dataset, then pass the result in here.
        """

        context = CSQAContext.read_from_file(kg_path=self.kg_path,
                                             kg_type_path=self.kg_type_path,
                                             entity_id2string_path=self.entity_id2string_path,
                                             predicate_id2string_path=self.predicate_id2string_path,
                                             question_type=question_type,
                                             question_tokens=None,
                                             question_entities=question_entities,
                                             question_predicates=question_predicates,
                                             question_type_entities=question_type_entities)

        if self.use_bert_encoder:
            question = prepare_question_for_bert(question,
                                                 question_entities,
                                                 question_type_entities,
                                                 question_predicates,
                                                 context,
                                                 self.augment_sentence_with_context)
        else:
            question = question.lower()

        tokenized_question = self.tokenizer.tokenize(question)

        question_field = TextField(tokenized_question, self.question_token_indexers)
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question], "answer": answer}

        context.question_tokens = tokenized_question

        language = CSQALanguage(context)

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            # TODO: make difference between global and local rules
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)

        # Add empty rule (remove when loop above is implemented).
        action_field = ListField(production_rule_fields)
        # TODO add type_list_field?
        # type_list_field = MetadataField(question_type_entities)

        expected_result = parse_answer(answer, entities_result, language)
        if expected_result is None:
            return None

        expected_result_field = MetadataField(expected_result)
        result_entities_field = MetadataField([language.get_entity_from_question_id(e) for e in entities_result])

        fields = {'qa_id': MetadataField(qa_id),
                  'question_type': MetadataField(question_type),
                  'question_description': MetadataField(question_description),
                  'question': question_field,
                  'expected_result': expected_result_field,
                  'world': MetadataField(language),
                  'actions': action_field,
                  'metadata': MetadataField(metadata),
                  "result_entities": result_entities_field,
                  'question_predicates': MetadataField(question_predicates)}

        def create_action_sequences_field(logical_forms, action_map_):
            return ListField([ListField([IndexField(action_map_[a], action_field) for a in l]) for l in logical_forms])

        # TODO: Assuming that possible actions are the same in all worlds. This is not true
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
        target_action_sequences_field: ListField = create_action_sequences_field(qa_logical_forms, action_map)
        fields['target_action_sequences'] = target_action_sequences_field

        return Instance(fields)
