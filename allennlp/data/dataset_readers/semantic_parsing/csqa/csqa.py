
"""
Reader for CSQA (https://amritasaha1812.github.io/CSQA/).
"""

from typing import Dict, List, Any
import json
import logging
import os

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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        boolean indicating whether we only read direct questions (without references to questions earlier in the
        conversation)
    """
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
                 kg_type_data_path: str = None,
                 entity_id2string_path: str = None,
                 predicate_id2string_path: str = None,
                 skip_approximate_questions: bool = True,
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
        self.kg_type_data_path = kg_type_data_path
        self.predicate_id2string_path = predicate_id2string_path
        self.load_direct_questions_only = read_only_direct
        self.skip_approximate_questions = skip_approximate_questions
        self.shared_kg_context = None

    def get_empty_context(self):
        return CSQAContext.read_from_file(self.kg_path, self.kg_type_data_path, self.entity_id2string_path,
                                          self.predicate_id2string_path, [], [], [], [])

    @overrides
    def _read(self, qa_path: str):
        if qa_path.endswith('.json'):
            file_id = 'sample'
            yield from self._read_unprocessed_file(qa_path, file_id)

        # read from data directory
        elif os.path.isdir(qa_path):
            qa_path = Path(qa_path)
            for file_path in qa_path.glob('**/*.json'):
                # if our file_path is some_dir/some_other_dir/train/QA_0/QA_0.json,
                # the file_id is train/QA_0/QA_0.json
                file_id = file_path.relative_to(qa_path.parent)
                yield from self._read_unprocessed_file(file_path, str(file_id))
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {qa_path}")

    def _read_unprocessed_file(self, qa_file_path: str, file_id: str):
        # initialize a "shared context" object, which we only create once as it is very expensive to read the kg, and
        # we can re-use the kg for each object
        if not self.shared_kg_context:
            self.shared_kg_context = CSQAContext.read_from_file(self.kg_path,
                                                                self.kg_type_data_path,
                                                                self.entity_id2string_path,
                                                                self.predicate_id2string_path,
                                                                [], [], [], [])

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
                question_entities = qa_dict_question["entities_in_utterance"]
                question_predicates = qa_dict_question["relations"]
                question_type_list = qa_dict_question["type_list"]
                question_description = qa_dict_question["description"]
                question_type = qa_dict_question["question-type"]
                answer = qa_dict_answer["utterance"]
                answer_description = qa_dict_answer["description"]
                entities_result = qa_dict_answer["all_entities"]

                # TODO: do we need extra checks here (e.g. 2 clarifications in a row)?
                if skip_next_turn:
                    skip_next_turn = False
                    continue

                if self.load_direct_questions_only:
                    # If this question requires clarification, we skip the next (as it will contain a reference)
                    if "Clarification" in answer_description:
                        skip_next_turn = True
                    if "Indirect" in question_description or "indirect" in question_description or "indirectly" in \
                            question_description or "Incomplete" in question_description or "Coreferenced" in \
                            question_type or "Ellipsis" in question_type:
                        continue

                if self.skip_approximate_questions:
                    if "approximate" in question:
                        continue

                # TODO: implement reading dynamic programming denotations
                if self._dpd_output_directory:
                    pass
                else:
                    logical_forms = None

                instance = self.text_to_instance(qa_id=qa_id,
                                                 question_type=question_type,
                                                 question_description=question_description,
                                                 question=question,
                                                 question_entities=question_entities,
                                                 question_predicates=question_predicates,
                                                 answer=answer,
                                                 entities_result=entities_result,
                                                 type_list=question_type_list,
                                                 kg_data=kg_data,
                                                 kg_type_data=kg_type_data,
                                                 entity_id2string=entity_id2string,
                                                 predicate_id2string=predicate_id2string,
                                                 dpd_output=logical_forms)
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
                         type_list: List[str],
                         kg_data: List[Dict[str, str]],
                         kg_type_data: List[Dict[str, str]],
                         entity_id2string: Dict[str, str],
                         predicate_id2string: Dict[str, str],
                         dpd_output: List[str] = None,
                         tokenized_question: List[Token] = None) -> object:
        """
        Reads text inputs and makes an instance.
        Parameters
        ----------
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
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question], "answer": answer}

        context = CSQAContext.read_from_file('', '', '', '',
                                             tokenized_question,
                                             question_entities,
                                             question_predicates,
                                             kg_data=kg_data,
                                             kg_type_data=kg_type_data,
                                             type_list=type_list,
                                             entity_id2string=entity_id2string, predicate_id2string=predicate_id2string)
        language = CSQALanguage(context)

        production_rule_fields: List[Field] = []
        for production_rule in language.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)

        # Add empty rule (remove when loop above is implemented).
        action_field = ListField(production_rule_fields)
        qa_id_field = MetadataField(qa_id)
        world_field = MetadataField(language)
        predicate_field = MetadataField(question_predicates)
        type_list_field = MetadataField(type_list)

        # parse answer
        if answer in ["YES", "NO"]:
            expected_result = True if answer is "YES" else False
        elif entities_result:
            # check if result is a count of entities
            try:
                expected_result = int(answer)
            # read entities in result
            except ValueError:
                # expected_result = {Entity(ent, ent) for ent in entities_result}
                expected_result = {language.get_entity_from_question_id(ent) for ent in entities_result}
        elif answer.startswith("Did you mean"):
            expected_result = "clarification"
        elif answer == "YES and NO respectively" or answer == "NO and YES respectively":
            return None
        else:
            raise ValueError("unknown answer format: {}".format(answer))

        expected_result_field = MetadataField(expected_result)

        if entities_result:
            result_entities_field = ListField([LabelField(result_entity, label_namespace='denotations')
                                               for result_entity in entities_result])
        else:
            result_entities_field = ListField([LabelField("none")])

        # 'answer': answer_field,
        fields = {'qa_id': qa_id_field,
                  'question_type': question_type,
                  'question_description': question_description,
                  'question': question_field,
                  'expected_result': expected_result_field,
                  'world': world_field,
                  'actions': action_field,
                  'metadata': MetadataField(metadata),
                  "result_entities": result_entities_field,
                  'question_predicates': predicate_field}

        # TODO: Assuming that possible actions are the same in all worlds. This is not true of course
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

        # TODO: implement this part
        if dpd_output:
            action_sequence_fields: List[Field] = []
            fields['target_action_sequences'] = ListField(action_sequence_fields)
            # TODO add correct indices
        else:
            # Create fake programs if not provided. Creates a dummy query that gets the last entity provided by the
            # question. If no entity is present in the question, it returns to skip this instance.
            entity = [prod for prod in language.all_possible_productions() if prod.startswith('Entity ->')]
            if not entity:
                return
            productions = ['@start@ -> Set[Entity]', 'Set[Entity] -> [<Entity:Set[Entity]>, Entity]',
                           '<Entity:Set[Entity]> -> get', entity[-1]]
            indices = ([language.all_possible_productions().index(prod) for prod in productions])
            action_sequence_fields: List[Field] = [ListField([IndexField(idx, action_field) for idx in indices])]
            fields['target_action_sequences'] = ListField(action_sequence_fields)

        return Instance(fields)
