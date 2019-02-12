from typing import Dict, List, Tuple
from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.models.model import Model
from allennlp.semparse.domain_languages import CSQALanguage
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import Average


class CSQASemanticParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super(CSQASemanticParser, self).__init__(vocab=vocab)

        self._sentence_embedder = sentence_embedder
        self._denotation_accuracy = Average()
        self._consistency = Average()
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

    @overrides
    def forward(self):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(self, sentence: Dict[str, torch.LongTensor]):
        raise NotImplementedError

    def _create_grammar_state(self,
                              world: CSQALanguage,
                              possible_actions: List[ProductionRule]) -> GrammarStatelet:
        raise NotImplementedError

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. We only transform the action string sequences into logical
        forms here.
        """
        raise NotImplementedError
