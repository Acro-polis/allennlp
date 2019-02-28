# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts import CSQADgraphContext
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.ip = 'localhost:9080'
        self.entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        self.predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    def test_kg_data(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_dgraph_context = CSQADgraphContext.connect_to_db(self.ip, self.entity_id2string_path,
                                                              self.predicate_id2string_path, question_tokens,
                                                              question_entities)

        # Run query.
        query = """
        { test1(func: allofterms(wikidata_uid, "Q15140125")) { expand(_all_) {expand(_all_) } } }
        { test2(func: allofterms(wikidata_uid, "Q1253489")) { expand(_all_) {expand(_all_) } } }
        { test3(func: allofterms(wikidata_uid, "Q1253486")) { expand(_all_) {expand(_all_) } } }
        { test4(func: allofterms(wikidata_uid, "Q12122755")) { expand(_all_) {expand(_all_) } }
        } { test5(func: allofterms(wikidata_uid, "Q274244")) { expand(_all_) {expand(_all_) } } }
        """

        res = csqa_dgraph_context.kg_data.query(query)
        res_json = json.loads(res.json)
        res_expected = {"test1": [{"wikidata_uid": "Q15140125",
                                   "P31": [{"wikidata_uid": "Q20010800"}]}],
                        "test2": [{"wikidata_uid": "Q1253489",
                                   "P27": [{"wikidata_uid": "Q262"}, {"wikidata_uid": "Q142"}],
                                   "P19": [{"wikidata_uid": "Q262"}],
                                   "P31": [{"wikidata_uid": "Q5"}],
                                   "P69": [{"wikidata_uid": "Q1664782"}],
                                   "P106": [{"wikidata_uid": "Q7042855"}, {"wikidata_uid": "Q2526255"},
                                            {"wikidata_uid": "Q1415090"}, {"wikidata_uid": "Q28389"}],
                                   "P735": [{"wikidata_uid": "Q1605665"}],
                                   "P21": [{"wikidata_uid": "Q6581097"}]}],
                        "test3": [{"wikidata_uid": "Q1253486",
                                   "P17": [{"wikidata_uid": "Q159"}],
                                   "P31": [{"wikidata_uid": "Q2198484"}],
                                   "P36": [{"wikidata_uid": "Q141697"}],
                                   "P131": [{"wikidata_uid": "Q5207"}]}],
                        "test4": [{"wikidata_uid": "Q12122755",
                                   "P161": [{"wikidata_uid": "Q254431"}, {"wikidata_uid": "Q110374"},
                                            {"wikidata_uid": "Q495487"}, {"wikidata_uid": "Q467519"},
                                            {"wikidata_uid": "Q296505"}],
                                   "P364": [{"wikidata_uid": "Q1860"}],
                                   "P31": [{"wikidata_uid": "Q11424"}],
                                   "P86": [{"wikidata_uid": "Q1900269"}],
                                   "P162": [{"wikidata_uid": "Q5049340"}, {"wikidata_uid": "Q475223"}],
                                   "P495": [{"wikidata_uid": "Q30"}],
                                   "P57": [{"wikidata_uid": "Q7342218"}],
                                   "P136": [{"wikidata_uid": "Q172980"}]}],
                        "test5": [{"wikidata_uid": "Q274244",
                                   "P21": [{"wikidata_uid": "Q6581072"}],
                                   "P1411": [{"wikidata_uid": "Q281939"}],
                                   "P166": [{"wikidata_uid": "Q281939"}],
                                   "P735": [{"wikidata_uid": "Q564684"}],
                                   "P119": [{"wikidata_uid": "Q1624932"}],
                                   "P27": [{"wikidata_uid": "Q30"}],
                                   "P19": [{"wikidata_uid": "Q38022"}],
                                   "P31": [{"wikidata_uid": "Q5"}],
                                   "P106": [{"wikidata_uid": "Q7042855"}],
                                   "P20": [{"wikidata_uid": "Q1337818"}]}]}
        assert res_json == res_expected

    def test_alphabetic_entity_extraction(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_context = CSQADgraphContext.connect_to_db(self.ip, self.entity_id2string_path,
                                                       self.predicate_id2string_path, question_tokens,
                                                       question_entities)

        entities, _ = csqa_context.get_entities_from_question()
        assert entities == ["Q12122755"]

    def test_number_extraction(self):
        question = "Which fictional characters had their voice dubbing done by atmost 3 people ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_context = CSQADgraphContext.connect_to_db(self.ip, self.entity_id2string_path,
                                                       self.predicate_id2string_path, question_tokens,
                                                       question_entities)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 10)]

    def test_date_extraction(self):
        question = "Which people took part in the March 2007 Iditarod and are a male ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_context = CSQADgraphContext.connect_to_db(self.ip, self.entity_id2string_path,
                                                       self.predicate_id2string_path, question_tokens,
                                                       question_entities)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 6), ("2007", 7)]

    def test_rank_number_extraction(self):
        question = "How many television stations were the first to air greater number of" \
                   " television programs or television genres than American Heroes Channel ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_context = CSQADgraphContext.connect_to_db(self.ip, self.entity_id2string_path,
                                                       self.predicate_id2string_path, question_tokens,
                                                       question_entities)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("1", 6)]
