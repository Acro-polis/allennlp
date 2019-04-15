import logging
from typing import Any, List, Dict

import torch
from collections import OrderedDict
from overrides import overrides

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.csqa.csqa_semantic_parser import CSQASemanticParser
from allennlp.modules import Attention, TextFieldEmbedder, KgEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from allennlp.semparse.domain_languages.csqa_language import CSQALanguage
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import BasicTransitionFunction
from allennlp.data.dataset_readers.semantic_parsing.csqa.csqa import COUNT_QUESTION_TYPES, OTHER_QUESTION_TYPES

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("csqa_mml_parser")
class CSQAMmlSemanticParser(CSQASemanticParser):
    """
    ``CSQAMMlSemanticParser`` is an ``CSQASemanticParser`` that solves the problem of lack of
    logical form annotations by maximizing the marginal likelihood of an approximate set of target
    sequences that yield the correct denotation. This parser takes the output of an offline search
    process as the set of target sequences for training, the latter performs search during training.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    action_embedding_dim : ``int``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the TransitionFunction.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        Maximum number of steps for beam search after training.
    dropout : ``float``, optional (default=0.0)
        Probability of dropout to apply on encoder outputs, decoder outputs and predicted actions.
    direct_questions_only : ``bool``, optional (default=True)
        Only train on direct question (i.e.: without questions that refer to earlier conversation).
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 kg_embedder: KgEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 dropout: float = 0.0,
                 direct_questions_only=True) -> None:
        super(CSQAMmlSemanticParser, self).__init__(vocab=vocab,
                                                    sentence_embedder=sentence_embedder,
                                                    kg_embedder=kg_embedder,
                                                    action_embedding_dim=action_embedding_dim,
                                                    encoder=encoder,
                                                    dropout=dropout,
                                                    direct_questions_only=direct_questions_only)

        self._decoder_trainer = MaximumMarginalLikelihood()
        self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=attention,
                                                     num_start_types=1,
                                                     activation=Activation.by_name('tanh')(),
                                                     predict_start_type_separately=False,
                                                     add_action_bias=False,
                                                     dropout=dropout)
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

    @overrides
    def forward(self,  # type: ignore
                qa_id,
                question: Dict[str, torch.LongTensor],
                question_type,  # TODO add types to arguments
                question_description,
                question_entities,
                question_type_entities,
                question_predicates,
                expected_result,
                world: List[CSQALanguage],
                actions: List[List[ProductionRule]],
                identifier: List[str] = None,
                target_action_sequences: torch.LongTensor = None,
                result_entities=None,
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences, trained to maximize marginal
        likelihood over a set of approximate logical forms.
        """
        # print(question['tokens'].size())
        batch_size = question['tokens'].size()[0]

        # TODO: embed entities
        if True:
            embedded_entities = self._kg_embedder(question_entities, input_type="entity")
            embedded_type_entities = self._kg_embedder(question_type_entities, input_type="entity")
            embedded_predicates = self._kg_embedder(question_predicates, input_type="predicate")

            # print(question_entities, question_type_entities, question_predicates)
            # print(embedded_entities, embedded_type_entities, embedded_predicates)

        initial_rnn_state = self._get_initial_rnn_state(question)
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float) for _ in range(batch_size)]

        result_entities: List[List[str]] = self._get_label_strings(result_entities) \
            if result_entities is not None else None

        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_statelet = [self._create_grammar_state(world[i], actions[i]) for i in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_statelet,
                                          possible_actions=actions,
                                          extras=result_entities)

        if target_action_sequences is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequences = target_action_sequences.squeeze(-1)
            target_mask = target_action_sequences != self._action_padding_index
        else:
            target_mask = None

        outputs: Dict[str, torch.Tensor] = {}
        # TODO: does this if statement make sense if it is overwritten by decoder_trainer.decode()?
        if identifier is not None:
            outputs["identifier"] = identifier
        if target_action_sequences is not None:
            outputs = self._decoder_trainer.decode(initial_state,
                                                   self._decoder_step,
                                                   (target_action_sequences, target_mask))

        if not self.training:
            # print(question_type)
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=False)
            best_action_sequences: Dict[int, List[List[int]]] = {}
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    best_action_indices = [best_final_states[i][0].action_history[0]]
                    best_action_sequences[i] = best_action_indices
            batch_action_strings = self._get_action_strings(actions, best_action_sequences)
            batch_denotations = self._get_denotations(batch_action_strings, world)

            if target_action_sequences is not None:
                self._update_metrics(action_strings=batch_action_strings,
                                     worlds=world,
                                     label_strings=result_entities,
                                     question_types=question_type)
            else:
                if metadata is not None:
                    outputs["sentence_tokens"] = [x["sentence_tokens"] for x in metadata]
                outputs['debug_info'] = []
                for i in range(batch_size):
                    outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
                outputs["best_action_strings"] = batch_action_strings
                outputs["denotations"] = batch_denotations
                action_mapping = {}
                for batch_index, batch_actions in enumerate(actions):
                    for action_index, action in enumerate(batch_actions):
                        action_mapping[(batch_index, action_index)] = action[0]
                outputs['action_mapping'] = action_mapping
        return outputs

    def _update_metrics(self,
                        action_strings: List[List[List[str]]],
                        worlds: List[CSQALanguage],
                        label_strings: List[List[str]],
                        question_types: List[str]) -> None:
        batch_size = len(worlds)

        for i in range(batch_size):
            instance_action_strings = action_strings[i]
            # if instance_action_strings:
            instance_label_strings = label_strings[i]
            instance_world = worlds[i]
            question_type = question_types[i]
            # Taking only the best sequence.
            if question_type in self.retrieval_question_types:
                precision_metric = self._metrics[question_type + " precision"]
                recall_metric = self._metrics[question_type + " recall"]
                if instance_action_strings:
                    retrieved_entities = self._get_retrieved_entities(instance_action_strings[0],
                                                                      instance_world)
                else:
                    retrieved_entities = []

                precision_metric(instance_label_strings, retrieved_entities)
                recall_metric(instance_label_strings, retrieved_entities)

            elif question_type in OTHER_QUESTION_TYPES + COUNT_QUESTION_TYPES:
                metric = self._metrics[question_type + " accuracy"]
                if instance_action_strings:
                    sequence_is_correct: bool = self._check_denotation(instance_action_strings[0],
                                                                       instance_label_strings,
                                                                       instance_world,
                                                                       question_type)
                else:
                    sequence_is_correct: bool = False
                metric(1 if sequence_is_correct else 0)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = OrderedDict()
        for key, metric in self._metrics.items():
            tensorboard_key = "_" + key
            # Tensorboard does not allow spaces and brackets in keys.
            for c in [" ", "(", ")"]:
                tensorboard_key = tensorboard_key.replace(c, "_")
            if not self.training:
                metrics[tensorboard_key] = metric.get_metric(reset)
            else:
                # Also write to dict when not tracking these metrics (for train) to preserve
                # metric ordering when printing.
                metrics[tensorboard_key] = None
        return metrics
