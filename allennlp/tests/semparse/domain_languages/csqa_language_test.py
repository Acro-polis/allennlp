# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestCSQALanguage(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # Adding a bunch of random tokens in here so we get them as constants in the language.
        self.kg_test_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        self.entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        self.predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

        question = "which administrative territory is the country of origin of frank and jesse ?"
        self.question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]

        self.csqa_context = CSQAContext.read_from_file(self.kg_test_path, self.entity_id2string_path,
                                                       self.predicate_id2string_path, self.question_tokens,
                                                       question_entities)
        self.language = CSQALanguage(self.csqa_context)

    def _get_world_with_question_tokens_and_entities(self,
                                                     question_tokens: List[Token],
                                                     question_entities: List[str]) -> CSQALanguage:
        csqa_context = CSQAContext.read_from_file(self.kg_test_path, self.entity_id2string_path,
                                                  self.predicate_id2string_path, question_tokens, question_entities)
        language = CSQALanguage(csqa_context)
        return language

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows Q12122755)"
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_get(self):
        logical_form = "(get Q12122755)"
        entity_list = self.language.execute(logical_form)
        assert entity_list == ["Q12122755"]

    def test_execute_works_with_find(self):
        logical_form = "(find all_entities P495)"
        entity_list = self.language.execute(logical_form)
        assert entity_list == ["Q30"]

    def test_execute_works_with_count(self):
        logical_form = "(count (find all_entities P161))"
        count = self.language.execute(logical_form)
        assert count == 5

    def test_is_in(self):
        question_entities = ["Q12122755", "Q254431"]
        language = self._get_world_with_question_tokens_and_entities(self.question_tokens, question_entities)
        logical_form = "(is_in Q254431 (find all_entities P161))"
        boolean = language.execute(logical_form)
        assert boolean

    def test_union(self):
        logical_form = "(union (find all_entities P495) (find all_entities P161))"
        entity_list = self.language.execute(logical_form)
        assert set(entity_list) == {"Q30", "Q254431", "Q110374", "Q495487", "Q467519", "Q296505"}

    def test_intersection(self):
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(self.question_tokens, question_entities)
        logical_form = "(intersection (find (get Q274244) P106) (find (get Q1253489) P106))"
        assert language.execute(logical_form) == ["Q7042855"]

    def test_diff(self):
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(self.question_tokens, question_entities)
        logical_form = "(diff (find (get Q1253489) P106) (find (get Q274244) P106))"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {"Q2526255", "Q1415090", "Q28389"}

    def test_larger(self):
        question = "which american president has more than 2 brothers?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(question_tokens, question_entities)
        logical_form = "(larger all_entities P106 2)"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {"Q1253489"}

    def test_smaller(self):
        question = "which american president has less than 2 brothers?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(question_tokens, question_entities)
        logical_form = "(less all_entities P106 2)"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {'Q15140125', 'Q12122755', 'Q274244', 'Q1253486'}

    def test_equal(self):
        question = "which american president has exactly 4 brothers?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(question_tokens, question_entities)
        logical_form = "(equal all_entities P106 4)"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {"Q1253489"}

    def test_most(self):
        question = "which american president has at most than 4 brothers?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(question_tokens, question_entities)
        logical_form = "(most all_entities P106 4)"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {"Q1253489", 'Q15140125', 'Q12122755', 'Q274244', 'Q1253486'}

    def test_least(self):
        question = "which american president has at least than 1 brothers?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q274244", "Q1253489"]
        language = self._get_world_with_question_tokens_and_entities(question_tokens, question_entities)
        logical_form = "(least all_entities P106 1)"
        entity_set = set(language.execute(logical_form))
        assert entity_set == {'Q274244', 'Q1253489'}

    # TODO: test union empty set
    # TODO: test intersection empty set
    # TODO: test instantiate entity
    # TODO: test instantiate predicate
