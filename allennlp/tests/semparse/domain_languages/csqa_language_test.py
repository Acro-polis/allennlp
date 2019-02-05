# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import List

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages.domain_language import ExecutionError
from allennlp.semparse.domain_languages.wikitables_language import Date, WikiTablesLanguage
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class TestCSQALanguage(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        # Adding a bunch of random tokens in here so we get them as constants in the language.

        question_tokens = [Token(x) for x in["which", "administrative", "territory", "is", "the", "country", "of",
                           "origin", "of", "frank", "and", "jesse", "?"]]

        self.table_file = self.FIXTURES_ROOT / 'data' / 'wikitables' / 'sample_table.tagged'
        self.table_context = TableQuestionContext.read_from_file(self.table_file, question_tokens)
        self.language = WikiTablesLanguage(self.table_context)
