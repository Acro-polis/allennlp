# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import CSQADatasetReader
from allennlp.semparse.domain_languages import CSQALanguage


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 17
    instance = instances[0]

    assert instance.fields.keys() == {
        'question',
        'answer',
        'world',
        'actions',
        'metadata',
        'target_action_sequences'
    }

    language = instance.fields['world'].as_tensor({})
    question_tokens = ["which", "administrative", "territory", "is", "the", "country", "of",
                       "origin", "of", "frank", "and", "jesse", "?"]

    assert [t.text for t in instance.fields["question"].tokens] == question_tokens
    assert isinstance(language, CSQALanguage)

    action_fields = instance.fields['actions'].field_list
    first_action_sequence = instance.fields["target_action_sequences"].field_list[0]

    actions_vocab = [action_field.rule for action_field in action_fields]
    action_indices = [l.sequence_index for l in first_action_sequence.field_list]
    actions = [actions_vocab[i] for i in action_indices]
    assert actions == ['@start@ -> List[str]', '<str:List[str]> -> get', 'str -> Q12122755']


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
