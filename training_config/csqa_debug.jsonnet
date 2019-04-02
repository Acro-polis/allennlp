{
  "dataset_reader": {
    "type": "csqa",
    "kg_path": "allennlp/tests/fixtures/data/csqa/sample_kg.p",
    "kg_type_path": "allennlp/tests/fixtures/data/csqa/sample_par_child_dict.p",
    "dpd_output_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000_logical_forms.p",
    "entity_id2string_path": "allennlp/tests/fixtures/data/csqa/sample_entity_id2string.json",
    "predicate_id2string_path": "allennlp/tests/fixtures/data/csqa/filtered_property_wikidata4.json",
    "read_only_direct": true,
    "lazy": true
  },
  "vocabulary": {
    "non_padded_namespaces": ["denotations", "rule_labels"]
  },
 "train_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000.tar.gz",
//  "train_data_path": "allennlp/tests/fixtures/data/csqa/sample_train",
 "validation_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_valid_1100.tar.gz",
//  "validation_data_path": "allennlp/tests/fixtures/data/csqa/sample_train_7000.tar.gz",
//  "validation_data_path": "allennlp/tests/fixtures/data/csqa/sample_train",
//  "validation_data_path": "/home/ubuntu/Desktop/CSQA_v9/test.tar.gz",
//  "validation_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/valid.tar.gz",

  "model": {
    "type": "csqa_mml_parser",
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 25,
          "trainable": true
        }
      }
    },
    "direct_questions_only": true,
    "action_embedding_dim": 50,
    "encoder": {
      "type": "lstm",
      "input_size": 25,
      "hidden_size": 10,
      "num_layers": 1
    },
    "decoder_beam_search": {
      "beam_size": 5
    },
    "max_decoding_steps": 20,
    "attention": {"type": "dot_product"},
    "dropout": 0.0
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["question", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}
