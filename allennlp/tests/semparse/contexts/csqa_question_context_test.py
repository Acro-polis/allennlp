# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts import CSQAContext
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    def test_kg_data(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        csqa_context = CSQAContext.read_from_file(test_file, question_tokens)
        print(csqa_context.kg_data)
        # assert csqa_context.kg_data == [{'date_column:year': '2001',
        #                                               'number_column:division': '2',
        #                                               'string_column:league': 'usl_a_league',
        #                                               'string_column:regular_season': '4th_western',
        #                                               'string_column:playoffs': 'quarterfinals',
        #                                               'string_column:open_cup': 'did_not_qualify',
        #                                               'number_column:avg_attendance': '7_169'},
        #                                              {'date_column:year': '2005',
        #                                               'number_column:division': '2',
        #                                               'string_column:league': 'usl_first_division',
        #                                                'string_column:regular_season': '5th',
        #                                               'string_column:playoffs': 'quarterfinals',
        #                                               'string_column:open_cup': '4th_round',
        #                                               'number_column:avg_attendance': '6_028'}]


