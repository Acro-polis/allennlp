{
  "dataset_reader": {
    "type": "csqa",
    "kg_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/wikidata_short_1_2_rev.p",
    "kg_type_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/par_child_dict_full.p",
    "dpd_output_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000_logical_forms.p",
    "entity_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/items_wikidata_n.p",
    "predicate_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/filtered_property_wikidata4.json",
    "lazy": true,
    "augment_sentence_with_context": true,
    "tokenizer": {
       "type": "word",
       "word_splitter": {
          "type": "bert-basic"
       }
     },
    "question_token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "max_pieces": 128,
       }
    }
  },
  "validation_dataset_reader": {
    "type": "csqa",
    "kg_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/wikidata_short_1_2_rev.p",
    "kg_type_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/par_child_dict_full.p",
    "dpd_output_file": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_valid_1100_logical_forms.p",
    "entity_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/items_wikidata_n.p",
    "predicate_id2string_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/filtered_property_wikidata4.json",
    "read_only_direct": true,
    "lazy": true,
    "augment_sentence_with_context": true,
    "tokenizer": {
       "type": "word",
       "word_splitter": {
          "type": "bert-basic"
       }
     },
    "question_token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "max_pieces": 128,
       }
    }
  },
  "vocabulary": {
    "non_padded_namespaces": ["denotations", "rule_labels"]
  },
  "train_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_train_7000.tar.gz",
  "validation_data_path": "https://s3-eu-west-1.amazonaws.com/polisallennlp/datasets/CSQA/sample_valid_1100.tar.gz",
  "model": {
    "type": "csqa_mml_parser",
    "sentence_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets", "tokens_type_ids"],
        },
        "token_embedders": {
          "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": false,
            "top_layer_only": true
          },
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
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
