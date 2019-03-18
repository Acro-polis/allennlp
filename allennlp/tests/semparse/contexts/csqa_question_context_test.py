# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.contexts import CSQAContext
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class TestCSQAContext(AllenNlpTestCase):

    @classmethod
    def setUpClass(self):
        self.kg_test_path_json = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        self.kg_test_path_p = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.p'
        self.kg_type_test_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_par_child_dict.p'
        self.entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        self.predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        context = CSQAContext.read_from_file(self.kg_test_path_json,
                                             self.kg_type_test_path,
                                             self.entity_id2string_path,
                                             self.predicate_id2string_path)
        self.kg_data = context.kg_data
        self.kg_type_data = context.kg_type_data
        self.entity_id2string = context.entity_id2string
        self.predicate_id2string = context.predicate_id2string

    def setUp(self):
        super().setUp()

    def test_kg_data(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_predicates = ["P1"]
        type_list = ["Q1"]
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_json, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)

        assert csqa_context.kg_data == {"Q15140125": {"P31": ["Q20010800"]},
                                        "Q1253489": {"P2019": [], "P2387": [], "P2605": [], "P2604": [], "P214": [],
                                                     "P2519": [], "P19": ["Q262"], "P31": ["Q5"],
                                                     "P735": ["Q1605665"], "P2168": [], "P345": [], "P2639": [],
                                                     "P569": [], "P1266": [], "P2435": [], "P244": [],
                                                     "P69": ["Q1664782"],
                                                     "P106": ["Q7042855", "Q2526255", "Q1415090", "Q28389"],
                                                     "P227": [], "P27": ["Q142", "Q262"], "P21": ["Q6581097"],
                                                     "P1969": [], "P2626": []},
                                        "Q1253486": {"P721": [], "P646": [], "P131": ["Q5207"], "P625": [],
                                                     "P163": [], "P237": [], "P910": [], "P94": [], "P373": [],
                                                     "P242": [], "P856": [], "P150": [], "P17": ["Q159"], "P18": [],
                                                     "P764": [], "P31": ["Q2198484"], "P36": ["Q141697"],
                                                     "P571": [], "P41": [], "P214": []},
                                        "Q12122755": {"P136": ["Q172980"], "P1874": [], "P345": [], "P577": [],
                                                      "P162": ["Q5049340", "Q475223"],
                                                      "P161": ["Q254431", "Q110374", "Q495487", "Q467519",
                                                               "Q296505"], "P1562": [], "P3138": [],
                                                      "P86": ["Q1900269"], "P495": ["Q30"], "P364": ["Q1860"],
                                                      "P1237": [], "P2603": [], "P31": ["Q11424"], "P1258": [],
                                                      "P57": ["Q7342218"]},
                                        "Q274244": {"P646": [], "P2019": [], "P2387": [], "P2604": [], "P213": [],
                                                    "P2626": [], "P570": [], "P535": [], "P119": ["Q1624932"],
                                                    "P19": ["Q38022"], "P31": ["Q5"], "P735": ["Q564684"],
                                                    "P2168": [], "P345": [], "P1411": ["Q281939"], "P2639": [],
                                                    "P569": [], "P1266": [], "P2435": [], "P244": [],
                                                    "P106": ["Q7042855"], "P166": ["Q281939"], "P27": ["Q30"],
                                                    "P21": ["Q6581072"], "P20": ["Q1337818"], "P214": []}}

        csqa_context = CSQAContext.read_from_file(self.kg_test_path_p, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)

        assert csqa_context.kg_data == {15140125: {31: [20010800]},
                                        1253489: {19: [262], 31: [5], 735: [1605665], 69: [1664782],
                                                  106: [7042855, 2526255, 1415090, 28389], 27: [142, 262],
                                                  21: [6581097], },
                                        1253486: {131: [5207], 17: [159], 31: [2198484], 36: [141697]},
                                        12122755: {136: [172980], 162: [5049340, 475223],
                                                   161: [254431, 110374, 495487, 467519, 296505], 86: [1900269],
                                                   495: [30], 364: [1860], 31: [11424], 57: [7342218]},
                                        274244: {119: [1624932], 19: [38022], 31: [5], 735: [564684],
                                                 1411: [281939], 106: [7042855], 166: [281939], 27: [30],
                                                 21: [6581072], 20: [1337818]}}

    def test_alphabetic_entity_extraction(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_predicates = ["P1"]
        type_list = ["Q1"]
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_json, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        entities, _ = csqa_context.get_entities_from_question()
        assert entities == ["Q12122755"]

        question_entities = [12122755]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_p, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        entities, _ = csqa_context.get_entities_from_question()
        assert entities == [12122755]

    def test_number_extraction(self):
        question = "Which fictional characters had their voice dubbing done by atmost 3 people ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_predicates = ["P1"]
        type_list = ["Q1"]
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_json, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 10)]

        question_entities = [12122755]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_p, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 10)]

    def test_date_extraction(self):
        question = "Which people took part in the March 2007 Iditarod and are a male ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_entities = ["Q12122755"]
        question_predicates = ["P1"]
        type_list = ["Q1"]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_json, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 6), ("2007", 7)]

        question_entities = [12122755]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_p, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)
        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("3", 6), ("2007", 7)]

    def test_rank_number_extraction(self):
        question = "How many television stations were the first to air greater number of" \
                   " television programs or television genres than American Heroes Channel ?"
        question_tokens = self.tokenizer.tokenize(question)
        question_predicates = ["P1"]
        type_list = ["Q1"]
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_json, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)

        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("1", 6)]

        question_entities = [12122755]
        csqa_context = CSQAContext.read_from_file(self.kg_test_path_p, "", "", "",
                                                  question_predicates=question_predicates,
                                                  question_type_entities=type_list,
                                                  question_tokens=question_tokens,
                                                  question_entities=question_entities,
                                                  kg_type_data=self.kg_type_data,
                                                  entity_id2string=self.entity_id2string,
                                                  predicate_id2string=self.predicate_id2string)

        _, number_entities = csqa_context.get_entities_from_question()
        assert number_entities == [("1", 6)]
