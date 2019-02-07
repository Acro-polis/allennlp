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
        kg_test_file = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(kg_test_file, entity_id2string_path, predicate_id2string_path,
                                                  question_tokens, question_entities)
        assert csqa_context.kg_data == [{"P31": ["Q20010800"]},
                                        {"P2019": [], "P2387": [], "P2605": [], "P2604": [], "P214": [], "P2519": [],
                                         "P19": ["Q262"], "P31": ["Q5"], "P735": ["Q1605665"], "P2168": [], "P345": [],
                                         "P2639": [], "P569": [], "P1266": [], "P2435": [], "P244": [],
                                         "P69": ["Q1664782"], "P106": ["Q7042855", "Q2526255", "Q1415090", "Q28389"],
                                         "P227": [], "P27": ["Q142", "Q262"], "P21": ["Q6581097"], "P1969": [],
                                         "P2626": []},
                                        {"P721": [], "P646": [], "P131": ["Q5207"], "P625": [], "P163": [], "P237": [],
                                         "P910": [], "P94": [], "P373": [], "P242": [], "P856": [], "P150": [],
                                         "P17": ["Q159"], "P18": [], "P764": [], "P31": ["Q2198484"],
                                         "P36": ["Q141697"], "P571": [], "P41": [], "P214": []}]

    def test_alphabetic_entity_extraction(self):
        question = "which administrative territory is the country of origin of frank and jesse ?"
        question_tokens = self.tokenizer.tokenize(question)
        kg_test_file = f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json'
        entity_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json'
        predicate_id2string_path = f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        question_entities = ["Q12122755"]
        csqa_context = CSQAContext.read_from_file(kg_test_file, entity_id2string_path, predicate_id2string_path,
                                                  question_tokens, question_entities)
        assert csqa_context.question_entities == ["Q12122755"]

    def test_number_extraction(self):
        question = "Which fictional characters had their voice dubbing done by atmost 3 people ?"
        question_tokens = self.tokenizer.tokenize(question)



    # TODO: implement entity extraction (read from qa file)
    # TODO: implement relation extraction?
    # TODO: reuse number extraction code
    # TODO: pass entities to Language
