# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import CSQAContext
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage


class TestCSQALanguage(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # Adding a bunch of random tokens in here so we get them as constants in the language.

        question_tokens = [Token(x) for x in["which", "administrative", "territory", "is", "the", "country", "of",
                           "origin", "of", "frank", "and", "jesse", "?"]]

        self.kg_file = self.FIXTURES_ROOT / 'data' / 'csqa' / 'sample_kg.json'
        self.csqa_context = CSQAContext.read_from_file(self.kg_file, question_tokens)

        #TODO: implement
        # self.language = CSQALanguage(self.csqa_context)

    def _get_world_with_question_tokens(self, tokens: List[Token]) -> CSQALanguage:
        table_context = CSQAContext.read_from_file(self.table_file, tokens)
        world = CSQALanguage(table_context)
        return world

    def test(self):
        pass
