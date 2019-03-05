# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List
import time

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import CSQADgraphContext
from allennlp.semparse.domain_languages.csqa_dgraph_language import CSQADgraphLanguage
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestCSQADgraphLanguage(AllenNlpTestCase):

    @classmethod
    def setUpClass(self):

        self.ip = 'localhost:9080'
        # (un)comment these line to test the pickle file (with integer ids instead of string ids) or the full wikidata
        # self.kg_test_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        self.kg_test_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.p'
        # self.kg_test_path = f'{str(Path.home())}/Desktop/wikidata/wikidata_short_1_2.p'

        self.entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        self.predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'

        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self.question_entities = ["Q12122755", "Q274244", "Q1253489", "Q15140125", "Q1253489"]
        self.question = "which administrative territory is the country of origin of frank and jesse ?"
        self.question_tokens = self.tokenizer.tokenize(self.question)
        self.type_list = []
        self.context = CSQADgraphContext.connect_to_db(self.ip,
                                                       self.entity_id2string_path,
                                                       self.predicate_id2string_path,
                                                       self.question_tokens,
                                                       self.question_entities)
        self.language = CSQADgraphLanguage(self.context)

    def setUp(self):
        super().setUp()
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

    def _get_world_with_question_tokens_and_entities(self,
                                                     question_tokens: List[Token],
                                                     question_entities: List[str]) -> CSQADgraphLanguage:
        csqa_dgraph_context = CSQADgraphContext.connect_to_db("", "", "",
                                                              question_tokens=question_tokens,
                                                              question_entities=question_entities,
                                                              kg_data=self.context.kg_data,
                                                              entity_id2string=self.context.entity_id2string,
                                                              predicate_id2string=self.context.predicate_id2string)
        language = CSQADgraphLanguage(csqa_dgraph_context)
        return language

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows Q12122755)"
        with pytest.raises(ExecutionError):
            self.language.execute(logical_form)

    def test_execute_works_with_get(self):
        logical_form = "(get Q12122755)"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q12122755"}

    def test_execute_works_with_find(self):
        logical_form = "(find (union (get Q274244) (get Q1253489)) P19)"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q38022", "Q262"}

    def test_execute_works_with_count(self):
        logical_form = "(count (union (union (get Q274244) (get Q1253489)) (get Q12122755)))"
        count = self.language.execute(logical_form)
        assert count == 3

    def test_is_in(self):
        logical_form = "(is_in Q274244 (union (union (get Q274244) (get Q1253489)) (get Q12122755)))"
        boolean = self.language.execute(logical_form)
        assert boolean

        logical_form = "(is_in Q15140125 (union (union (get Q274244) (get Q1253489)) (get Q12122755)))"
        boolean = self.language.execute(logical_form)
        assert not boolean

    def test_union(self):
        logical_form = "(union (union (get Q274244) (get Q1253489)) (union (get Q12122755) (get Q15140125)))"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q12122755", "Q274244", "Q1253489", "Q15140125"}

        logical_form = "(union (get Q1253489) (intersection (get Q12122755) (get Q15140125)))"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q1253489"}

    def test_intersection(self):
        logical_form = "(intersection (union (get Q274244) (get Q12122755)) (union (get Q12122755) (get Q15140125)))"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q12122755"}

    def test_diff(self):
        logical_form = "(diff (union (get Q274244) (get Q12122755)) (union (get Q12122755) (get Q15140125)))"
        entity_set = set([ent.name for ent in self.language.execute(logical_form)])
        assert entity_set == {"Q274244"}

    def test_larger(self):
        question_tokens = self.tokenizer.tokenize("which american president has more than 2 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(larger (union (get Q274244) (get Q1253489)) P106 2)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q1253489"}

    def test_smaller(self):
        question_tokens = self.tokenizer.tokenize("which american president has less than 2 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(less (union (get Q274244) (get Q1253489)) P106 2)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q274244"}

    def test_equal(self):
        question_tokens = self.tokenizer.tokenize("which american president has exactly 4 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(equal (union (get Q274244) (get Q1253489)) P106 4)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q1253489"}

    def test_most(self):
        question_tokens = self.tokenizer.tokenize("which american president has at most 1 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(most (union (get Q274244) (get Q1253489)) P106 1)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q274244"}

        question_tokens = self.tokenizer.tokenize("which american president has at most 3 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(most (union (get Q274244) (get Q1253489)) P106 3)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q274244"}

        question_tokens = self.tokenizer.tokenize("which american president has at most 4 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(most (union (get Q274244) (get Q1253489)) P106 4)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q274244", "Q1253489"}

    def test_least(self):
        question_tokens = self.tokenizer.tokenize("which american president has at least 1 brother?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(least (union (get Q274244) (get Q1253489)) P106 1)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q274244", "Q1253489"}

        question_tokens = self.tokenizer.tokenize("which american president has at least 2 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(least (union (get Q274244) (get Q1253489)) P106 2)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q1253489"}

        question_tokens = self.tokenizer.tokenize("which american president has at least 4 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(least (union (get Q274244) (get Q1253489)) P106 4)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == {"Q1253489"}

        question_tokens = self.tokenizer.tokenize("which american president has at least 5 brothers?")
        language = self._get_world_with_question_tokens_and_entities(question_tokens, self.question_entities)
        logical_form = "(least (union (get Q274244) (get Q1253489)) P106 5)"
        entity_set = set([ent.name for ent in language.execute(logical_form)])
        assert entity_set == set()
