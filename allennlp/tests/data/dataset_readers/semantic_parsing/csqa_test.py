# pylint: disable=invalid-name,no-self-use,protected-access
import time

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import CSQADatasetReader
from allennlp.semparse.domain_languages import CSQALanguage


class CSQADatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads_direct(self):
        # read direct questions only
        params = {
                'lazy': False,
                'kg_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json',
                'kg_type_data_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_par_child_dict.p',
                'entity_id2string_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
                'predicate_id2string_path': f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
                }
        reader = CSQADatasetReader.from_params(Params(params))
        qa_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_qa.json'
        dataset = reader.read(qa_path)
        assert_dataset_correct(dataset, n_instances=15)

    def test_reader_reads_multiple_files_direct(self):
        # read direct questions only
        params = {
            'lazy': True,
            'kg_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json',
            'kg_type_data_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_par_child_dict.p',
            'entity_id2string_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
            'predicate_id2string_path': f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        }
        reader = CSQADatasetReader.from_params(Params(params))
        qa_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_train'
        dataset = reader.read(qa_path)
        instances = list(dataset)
        assert len(instances) == 32

    def test_reader_reads_dpd(self):
        params = {
            'lazy': True,
            'dpd_output_file': f'{self.FIXTURES_ROOT}/data/csqa/sample_train_action_sequences.p',
            'kg_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json',
            'kg_type_data_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_par_child_dict.p',
            'entity_id2string_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
            'predicate_id2string_path': f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        }
        reader = CSQADatasetReader.from_params(Params(params))
        qa_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_train'
        dataset = reader.read(qa_path)
        instances = list(dataset)
        assert len(instances) == 24

    def test_reader_reads_indirect(self):
        # read both direct and indirect questions
        params = {
            'lazy': False,
            'read_only_direct': False,
            'kg_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_kg.json',
            'kg_type_data_path': f'{AllenNlpTestCase.FIXTURES_ROOT}/data/csqa/sample_par_child_dict.p',
            'entity_id2string_path':  f'{self.FIXTURES_ROOT}/data/csqa/sample_entity_id2string.json',
            'predicate_id2string_path': f'{self.FIXTURES_ROOT}/data/csqa/filtered_property_wikidata4.json'
        }
        reader = CSQADatasetReader.from_params(Params(params))
        qa_path = f'{self.FIXTURES_ROOT}/data/csqa/sample_qa.json'
        dataset = reader.read(qa_path)
        assert_dataset_correct(dataset, n_instances=22)


def assert_dataset_correct(dataset, n_instances=19):
    instances = list(dataset)
    assert len(instances) == n_instances
    instance = instances[0]
    print(instance.fields.keys())
    assert instance.fields.keys() == {
        'qa_id',
        'question_type',
        'question_description',
        'question',
        'world',
        'actions',
        'metadata',
        'target_action_sequences',
        'expected_result',
        'result_entities',
        'question_predicates'
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

    assert actions == ['@start@ -> Set[Entity]',
                       'Set[Entity] -> [<Entity:Set[Entity]>, Entity]',
                       '<Entity:Set[Entity]> -> get',
                       'Entity -> Q15617994']
