"""
Reader for CSQA (https://amritasaha1812.github.io/CSQA/).
"""

from typing import Dict, List, Any
import json
import logging
import os
import pickle
import tarfile
from allennlp.common import Tqdm

from overrides import overrides
from collections import defaultdict
from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, LabelField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage
from allennlp.data.dataset_readers.semantic_parsing.csqa.util import get_dummy_action_sequences, question_is_indirect
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


RETRIEVAL_QUESTION_TYPES_DIRECT = ["Simple Question (Direct)", "Logical Reasoning (All)",
                                   "Quantitative Reasoning (All)", "Comparative Reasoning (All)"]
RETRIEVAL_QUESTION_TYPES_INDIRECT = ["Simple Question (Coreferenced)", "Simple Question (Ellipsis)", "Clarification"]
COUNT_QUESTION_TYPES = ["Quantitative Reasoning (Count) (All)", "Comparative Reasoning (Count) (All)"]
OTHER_QUESTION_TYPES = ["Verification (Boolean) (All)"]


# noinspection PyTypeChecker
@DatasetReader.register("csqa")
class CSQADatasetReader(DatasetReader):
    """

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    dpd_output_directory : ``str``, optional
        Directory that contains all the gzipped dpd output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``).  This is required for training a model, but not required
        for prediction.
    max_dpd_logical_forms : ``int``, optional (default=10)
        We will use the first ``max_dpd_logical_forms`` logical forms as our target label.  Only
        applicable if ``dpd_output_directory`` is given.
    sort_dpd_logical_forms : ``bool``, optional (default=True)
        If ``True``, we will sort the logical forms in the DPD output by length before selecting
        the first ``max_dpd_logical_forms``.  This makes the data loading quite a bit slower, but
        results in better training data.
    max_dpd_tries : ``int``, optional
        Sometimes DPD just made bad choices about logical forms and gives us forms that we can't
        parse (most of the time these are very unlikely logical forms, because, e.g., it
        hallucinates a date or number from the table that's not in the question).  But we don't
        want to spend our time trying to parse thousands of bad logical forms.  We will try to
        parse only the first ``max_dpd_tries`` logical forms before giving up.  This also speeds up
        data loading time, because we don't go through the entire DPD file if it's huge (unless
        we're sorting the logical forms).  Only applicable if ``dpd_output_directory`` is given.
        Default is 20.
    keep_if_no_dpd : ``bool``, optional (default=False)
        If ``True``, we will keep instances we read that don't have DPD output.  If you want to
        compute denotation accuracy on the full dataset, you should set this to ``True``.
        Otherwise, your accuracy numbers will only reflect the subset of the data that has DPD
        output.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use for the questions. Will default to ``WordTokenizer()`` with Spacy's tagger
        enabled, as we use lemma matches as features for entity linking.
    question_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    kg_path: ``str``, optional
        Path to the knowledge graph file. We use this file to initialize our context
    entity_id2string_path ``str``, optional
        Path to the json file which maps entity id's to their string values
    predicate_id2string_path ``str``, optional
        Path to the json file which maps predicate id's to their string values
    read_only_direct ``bool``, optional
        boolean indicating whether we only read direct questions (without references to questions
        earlier in the conversation)
    """
    def __init__(self,
                 lazy: bool = False,
                 dpd_output_file: str = None,
                 max_dpd_logical_forms: int = 10,
                 sort_dpd_logical_forms: bool = True,
                 max_dpd_tries: int = 20,
                 keep_if_no_dpd: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 kg_path: str = None,
                 kg_type_data_path: str = None,
                 entity_id2string_path: str = None,
                 predicate_id2string_path: str = None,
                 skip_approximate_questions: bool = True,
                 read_only_direct: bool = True
                 ) -> None:
        super().__init__(lazy=lazy)
        self._dpd_output_file = dpd_output_file
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._sort_dpd_logical_forms = sort_dpd_logical_forms
        self._max_dpd_tries = max_dpd_tries
        self._keep_if_no_dpd = keep_if_no_dpd
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.entity_id2string_path = entity_id2string_path
        self.kg_path = kg_path
        self.kg_type_data_path = kg_type_data_path
        self.predicate_id2string_path = predicate_id2string_path
        self.load_direct_questions_only = read_only_direct
        self.skip_approximate_questions = skip_approximate_questions
        self.shared_kg_context = None
        self.dpd_logical_form_dict = None

    def init_shared_kg_context(self):
        if not self.shared_kg_context:
            self.shared_kg_context = CSQAContext.read_from_file(kg_path=self.kg_path,
                                                                kg_type_data_path=self.kg_type_data_path,
                                                                entity_id2string_path=self.entity_id2string_path,
                                                                predicate_id2string_path=self.predicate_id2string_path)

    def init_dpd_dict(self):
        if not self.dpd_logical_form_dict and self._dpd_output_file:
            with open(self._dpd_output_file, 'rb') as input_file:
                self.dpd_logical_form_dict = pickle.load(input_file)

    @staticmethod
    def parse_answer(answer, entities_result, language):
        if answer in ["YES", "NO"]:
            return True if answer is "YES" else False
        elif entities_result:
            # check if result is a count of entities
            try:
                return int(answer)
            # read entities in result
            except ValueError:
                # expected_result = {Entity(ent, ent) for ent in entities_result}
                return {language.get_entity_from_question_id(ent) for ent in entities_result}
        elif answer.startswith("Did you mean"):
            return "clarification"
        elif answer == "YES and NO respectively" or answer == "NO and YES respectively":
            return None
        else:
            raise ValueError("unknown answer format: {}".format(answer))

    @overrides
    def _read(self, path: str):
        path = cached_path(path)
        if path.endswith('.json'):
            file_id = 'sample.json'
            yield from self._read_unprocessed_file(path, file_id)
        elif os.path.isdir(path):
            yield from self._read_from_directory(path)
        elif tarfile.is_tarfile(path):
            qa_path_dir = path.split('.')[:-2][0] if path.split('.')[-1] == 'gz' else path + "_dir"
            if os.path.isdir(qa_path_dir):
                print("Target folder for extraction already exists: ", qa_path_dir)
                yield from self._read_from_directory(qa_path_dir)
            else:
                print("Extracting into directory: ", qa_path_dir)
                os.mkdir(qa_path_dir)
                tar = tarfile.open(path)
                for member in Tqdm.tqdm(tar.getmembers()):
                    tar.extract(member, path=qa_path_dir)
                tar.close()
                yield from self._read_from_directory(qa_path_dir)
        else:
            raise ConfigurationError(f"Don't know how to read file type of {path}")

    def _read_from_directory(self, directory):
        directory = Path(directory)
        for file_path in directory.glob('**/*.json'):
            # The file_id is of format: QA_*/QA_*.json.
            file_id = file_path.relative_to(directory)
            yield from self._read_unprocessed_file(file_path, str(file_id))

    def _read_unprocessed_file(self, qa_file_path: str, file_id: str):
        # Initialize a "shared context" object, which we only create once as it is very expensive to
        # read the kg, and we can re-use the kg for each object.
        self.init_shared_kg_context()
        self.init_dpd_dict()

        kg_data = self.shared_kg_context.kg_data
        kg_type_data = self.shared_kg_context.kg_type_data
        entity_id2string = self.shared_kg_context.entity_id2string
        predicate_id2string = self.shared_kg_context.predicate_id2string

        with open(qa_file_path) as f:
            data = json.load(f)
            data = iter(data)
            skip_next_turn = False
            turn_id = 0
            for qa_dict_question in data:
                qa_id = file_id + str(turn_id)
                turn_id += 1

                qa_dict_question = defaultdict(str, qa_dict_question)
                qa_dict_answer = defaultdict(str, next(data))
                question = qa_dict_question["utterance"]
                question_description = qa_dict_question["description"]
                question_type = qa_dict_question["question-type"]
                question_type_entities = qa_dict_question["type_list"]
                answer_description = qa_dict_answer["description"]
                question_entities = qa_dict_question["entities_in_utterance"]

                # TODO: do we need extra checks here?
                if skip_next_turn:
                    skip_next_turn = False
                    continue

                if self.load_direct_questions_only:
                    # If this question requires clarification, we skip the next (as it will contain a reference)
                    skip_next_turn = "Clarification" in answer_description
                    if question_is_indirect(question_description, question_type):
                        continue

                if self.skip_approximate_questions:
                    if "approximate" in question or "around" in question:
                        continue

                if self._dpd_output_file:
                    # TODO: fix difference between empty list and no key
                    if qa_id in self.dpd_logical_form_dict:
                        qa_logical_forms = self.dpd_logical_form_dict[qa_id]
                        if not qa_logical_forms:
                            continue
                    else:
                        continue
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
                                                 kg_data=kg_data,
                                                 kg_type_data=kg_type_data,
                                                 entity_id2string=entity_id2string,
                                                 predicate_id2string=predicate_id2string,
                                                 qa_logical_forms=qa_logical_forms)
                if instance:
                    yield instance

    def text_to_instance(self,
                         qa_id: str,
                         question_type: str,
                         question_description: str,
                         question: str,
                         question_entities: List[str],
                         question_predicates: List[str],
                         answer: str,
                         entities_result: List[str],
                         question_type_entities: List[str],
                         kg_data: List[Dict[str, str]],
                         kg_type_data: List[Dict[str, str]],
                         entity_id2string: Dict[str, str],
                         predicate_id2string: Dict[str, str],
                         qa_logical_forms: List[str] = None,
                         tokenized_question: List[Token] = None) -> object:
        """
        Reads text inputs and makes an instance.
        Parameters
        ----------
        question_type
        question_description
        question_type_entities
        qa_id: ``str``
            qa turn id, e.g. 'train/QA_0/QA_1.json/0'
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
        kg_type_data : ``List[Dict[str, str]]``
            Type graph
        type_list: ``str``
            Types occurring in question
        entity_id2string : ``Dice[str, str]``
            Mapping from entity ids to there string values
        predicate_id2string : ``Dict[str, str]``
            Mapping from predicate ids to there string values
        qa_logical_forms : List[str], optional
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        tokenized_question : ``List[Token]``, optional
            If you have already tokenized the question, you can pass that in here, so we don't
            duplicate that work.  You might, for example, do batch processing on the questions in
            the whole dataset, then pass the result in here.
        """

        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question], "answer": answer}

        context = CSQAContext.read_from_file('', '', '', '',
                                             question_type=question_type,
                                             question_tokens=tokenized_question,
                                             question_entities=question_entities,
                                             question_predicates=question_predicates,
                                             question_type_entities=question_type_entities,
                                             kg_data=kg_data,
                                             kg_type_data=kg_type_data,
                                             entity_id2string=entity_id2string,
                                             predicate_id2string=predicate_id2string)
        language = CSQALanguage(context)

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            # TODO: make difference between global and local rules
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)

        # Add empty rule (remove when loop above is implemented).
        action_field = ListField(production_rule_fields)
        type_list_field = MetadataField(question_type_entities)

        expected_result = self.parse_answer(answer, entities_result, language)
        if expected_result is None:
            return None

        expected_result_field = MetadataField(expected_result)
        result_entities_field = ListField([LabelField(result_entity, label_namespace='denotations') for result_entity in
                                           entities_result] if entities_result else [LabelField("none")])

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
