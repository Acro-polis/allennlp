# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import CSQADatasetReader
from allennlp.semparse.worlds import CSQAWorld


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 14
    instance = instances[0]

    assert instance.fields.keys() == {
        'question',
        'answer',
        'world',
        'actions',
        'metadata'
    }

    question_tokens = ["which", "administrative", "territory", "is", "the", "country", "of",
                       "origin", "of", "frank", "and", "jesse", "?"]

    assert [t.text for t in instance.fields["question"].tokens] == question_tokens
    assert isinstance(instance.fields['world'].as_tensor({}), CSQAWorld)

    print("#"*10000)

    # TODO: implement actions and action tests


class CSQADatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        params = {
                'lazy': False,
                'kg_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json',
                'entity_id2string_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
                'predicate_id2string_path': f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
                }
        reader = CSQADatasetReader.from_params(Params(params))
        qa_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_qa.json'
        dataset = reader.read(qa_path)
        assert_dataset_correct(dataset)
