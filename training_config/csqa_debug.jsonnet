{
  "dataset_reader": {
    "type": "csqa",
    "kg_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_kg.p",
    "kg_type_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_kg.p",
//    "kg_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/wikidata_short_1_2_rev.p",
//    "kg_type_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/par_child_dict_full.p",
    "dpd_output_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000_logical_forms.p",
    "entity_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/items_wikidata_n.p",
    "predicate_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/filtered_property_wikidata4.json",
    "read_only_direct": true,
    "lazy": true
//    "add_entities_to_sentence": true,
//    "tokenizer": {
//       "type": "word",
//       "word_splitter": {
//          "type": "bert-basic"
//       }
//     },
//    "question_token_indexers": {
//      "tokens": {
//        "type": "bert-pretrained",
//        "pretrained_model": "bert-base-uncased"
//       }
//    }
  },
  "validation_dataset_reader": {
    "type": "csqa",
    "kg_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_kg.p",
    "kg_type_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_kg.p",
//    "kg_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/wikidata_short_1_2_rev.p",
//    "kg_type_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/par_child_dict_full.p",
    "dpd_output_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_valid_1100_logical_forms.p",
    "entity_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/items_wikidata_n.p",
    "predicate_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/filtered_property_wikidata4.json",
    "read_only_direct": true,
    "lazy": true
//    "add_entities_to_sentence": true,
//    "tokenizer": {
//       "type": "word",
//       "word_splitter": {
//          "type": "bert-basic"
//       }
//     },
//    "question_token_indexers": {
//      "tokens": {
//        "type": "bert-pretrained",
//        "pretrained_model": "bert-base-uncased",
//        "max_pieces": 256,
//       }
//    }
  },
  "vocabulary": {
    "non_padded_namespaces": ["denotations", "rule_labels"]
  },
// "train_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000.tar.gz",
  "train_data_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_train",
  "validation_data_path": std.extVar('ALLENNLP') + "/allennlp/tests/fixtures/data/csqa/sample_train",
// "validation_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_valid_1100.tar.gz",
//  "validation_data_path": "allennlp/tests/fixtures/data/csqa/sample_train_7000.tar.gz",
//  "validation_data_path": "allennlp/tests/fixtures/data/csqa/sample_train",
//  "validation_data_path": "/home/ruben/git/polis/Desktop/CSQA_v9/test.tar.gz",
//  "validation_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/valid.tar.gz",

  "model": {
    "type": "csqa_mml_parser",
    "kg_embedder": {
      "type": "kg_embedding",
      "embedding_dim": 50,
      "num_entities": 20982733,
      "num_predicates": 594,
      "trainable": false,
      "entity_pretrained_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/entity2vec.p",
      "predicate_pretrained_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/relation2vec.p",
//      "entity_pretrained_file": "notebooks/entity2vec.p",
//      "predicate_pretrained_file": "notebooks/relation2vec.p",
      "entity2id_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/entity2id.txt",
      "predicate2id_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/relation2id.txt"
    },
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 25,
          "trainable": true
        }
      },
//    "sentence_embedder": {
//        "allow_unmatched_keys": true,
//        "embedder_to_indexer_map": {
//            "tokens": ["tokens", "tokens-offsets", "tokens_type_ids"],
////            "tokens": ["tokens", "tokens-offsets", "tokens_segments"],
//        },
//        "token_embedders": {
//          "tokens": {
//            "type": "bert-pretrained",
//            "pretrained_model": "bert-base-uncased",
//            "requires_grad": true
//          },
//        }
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
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}
