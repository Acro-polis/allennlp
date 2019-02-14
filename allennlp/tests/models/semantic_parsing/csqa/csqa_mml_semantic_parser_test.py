from allennlp.common.testing import ModelTestCase


class CSQAMmlSemanticParsingTest(ModelTestCase):
    def setUp(self):
        super(CSQAMmlSemanticParsingTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / "semantic_parsing" /
                          "csqa" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "csqa" / "sample_qa.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
