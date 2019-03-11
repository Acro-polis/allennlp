from typing import Dict, List, Tuple
from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.semparse.contexts import CSQAContext
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.semparse.domain_languages import CSQALanguage, START_SYMBOL
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import F1Measure, BooleanAccuracy, Average
from allennlp.training.metrics.average_precision import AveragePrecision
from allennlp.training.metrics.average_recall import AverageRecall


class CSQASemanticParser(Model):
    """
    ``CSQASemanticParser`` is a semantic parsing model built for the CSQA domain. This is an
    abstract class and does not have a ``forward`` method implemented. Classes that inherit from
    this class are expected to define their own logic depending on the kind of supervision they
    use.  Accordingly, they should use the appropriate ``DecoderTrainer``. This class provides some
    common functionality for things like defining an initial ``RnnStatelet``, embedding actions,
    evaluating the denotations of completed logical forms, etc.

    Parameters
    ----------
    vocab : ``Vocabulary``
    sentence_embedder : ``TextFieldEmbedder``
        Embedder for sentences.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    dropout : ``float``, optional (default=0.0)
        Dropout on the encoder outputs.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super(CSQASemanticParser, self).__init__(vocab=vocab)
        self._sentence_embedder = sentence_embedder
        self._encoder = encoder

        self.retrieval_question_types = ["Simple Question (Direct)",
                                         "Simple Question (Coreferenced)",
                                         "Simple Question (Ellipsis)",
                                         "Logical Reasoning (All)",
                                         "Quantitative Reasoning (Count) (All)",
                                         "Clarification"]
        self.other_question_types = ["Verification (Boolean) (All)",
                                     "Quantitative Reasoning (All)",
                                     "Comparative Reasoning (All)"]

        precision_metrics = [(qt + " precision", AveragePrecision()) for qt in self.retrieval_question_types]
        recall_metrics = [(qt + " recall", AverageRecall()) for qt in self.retrieval_question_types]
        average_metrics = [(qt, Average()) for qt in self.other_question_types]

        self._metrics = dict(precision_metrics + recall_metrics + average_metrics)

        # self._metrics = {
        #     "Simple Question (Direct) precision": AveragePrecision(),
        #     "Simple Question (Direct) recall": AverageRecall(),
        #     "Simple Question (Coreferenced) precision": AveragePrecision(),
        #     "Simple Question (Coreferenced) recall": AverageRecall(),
        #     "Simple Question (Ellipsis) precision": AveragePrecision(),
        #     "Simple Question (Ellipsis) recall": AverageRecall(),
        #     "Logical Reasoning (All) precision": AveragePrecision(),
        #     "Logical Reasoning (All) recall": AverageRecall(),
        #     "Quantitative Reasoning (Count) (All) precision": AveragePrecision(),
        #     "Quantitative Reasoning (Count) (All) recall": AverageRecall(),
        #     "Clarification precision": AveragePrecision(),
        #     "Clarification recall": AverageRecall(),
        #     "Verification (Boolean) (All)": Average(),
        #     "Quantitative Reasoning (All)": Average(),
        #     "Comparative Reasoning (All)": Average()
        # }
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace

        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=action_embedding_dim)

        # This is what we pass as input in the first step of decoding, when we don't have a previous action.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)

    @overrides
    def forward(self):  # type: ignore
        # pylint: disable=arguments-differ
        # Sub-classes should define their own logic here.
        raise NotImplementedError

    def _get_initial_rnn_state(self, question: Dict[str, torch.LongTensor]):
        """
        This function encodes the question and computes attention over each question token and the final state. Then,
        each instance in the batch, an RnnStatelet is computed using the encodings, an empty memory and a first action
        embedding.
        """
        embedded_input = self._sentence_embedder(question)
        # TODO: embed entities

        # (batch_size, sentence_length)
        sentence_mask = util.get_text_field_mask(question).float()

        batch_size = embedded_input.size(0)

        # (batch_size, sentence_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(embedded_input, sentence_mask))

        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             sentence_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())
        attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                     encoder_outputs, sentence_mask)
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 attended_sentence[i],
                                                 encoder_outputs_list,
                                                 sentence_mask_list))
        return initial_rnn_state

    def _create_grammar_state(self,
                              world: CSQALanguage,
                              possible_actions: List[ProductionRule]) -> GrammarStatelet:
        """
        This function creates a GrammarStateLet by computing the valid actions
        """
        valid_actions = world.get_nonterminal_productions()
        action_mapping = {}
        for i, action in enumerate(possible_actions):
            action_mapping[action[0]] = i
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.
            action_indices = [action_mapping[action_string] for action_string in action_strings]
            # All actions in NLVR are global actions.
            global_actions = [(possible_actions[index][2], index) for index in action_indices]

            # Then we get the embedded representations of the global actions.
            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0)
            global_input_embeddings = self._action_embedder(global_action_tensor)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_input_embeddings,
                                                       list(global_action_ids))
        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               world.is_nonterminal)

    def _get_label_strings(self, labels):
        # TODO (pradeep): Use an unindexed field for labels?
        labels_data = labels.detach().cpu()
        label_strings: List[List[str]] = []
        for instance_labels_data in labels_data:
            label_strings.append([])
            for label in instance_labels_data:
                label_int = int(label)
                if label_int == -1:
                    # Padding, because not all instances have the same number of labels.
                    continue
                label_strings[-1].append(self.vocab.get_token_from_index(label_int, "denotations"))
        return label_strings

    @classmethod
    def _get_action_strings(cls,
                            possible_actions: List[List[ProductionRule]],
                            action_indices: Dict[int, List[List[int]]]) -> List[List[List[str]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            batch_best_sequences = action_indices[i] if i in action_indices else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [[batch_actions[rule_id][0] for rule_id in sequence]
                              for sequence in batch_best_sequences]
            all_action_strings.append(action_strings)
        return all_action_strings

    @staticmethod
    def _get_denotations(action_strings: List[List[List[str]]],
                         world: List[CSQALanguage]) -> List[List[List[str]]]:
        all_denotations: List[List[List[str]]] = []
        for instance_world, instance_action_sequences in zip(world, action_strings):
            denotations: List[List[str]] = []
            for instance_action_strings in instance_action_sequences:
                if not instance_action_strings:
                    continue
                logical_form = instance_world.action_sequence_to_logical_form(instance_action_strings)
                instance_denotations: List[str] = []

                if instance_world is not None:
                    instance_denotations.append(str(instance_world.execute(logical_form)))

                denotations.append(instance_denotations)
            all_denotations.append(denotations)
        return all_denotations

    @staticmethod
    def _check_denotation(action_sequence: List[str],
                          result_entities: List[str],
                          world: CSQALanguage) -> List[bool]:
        is_correct = []
        logical_form = world.action_sequence_to_logical_form(action_sequence)
        denotation = world.execute(logical_form)

        # TODO: clean this code

        if isinstance(result_entities, list):
            result_entities = set(result_entities)

        if isinstance(denotation, list):
            if isinstance(result_entities, set):
                is_correct.append(set(denotation) == result_entities)
            else:
                is_correct.append(False)
        elif isinstance(denotation, set):
            if isinstance(result_entities, set):
                is_correct.append(denotation == result_entities)
            else:
                is_correct.append(False)
        # TODO: deal with other types of results (counts, verification etc.)
        else:
            is_correct.append(False)
        return is_correct

    @staticmethod
    def _get_retrieved_entities(action_sequence: List[str],
                                world: CSQALanguage) -> List[bool]:

        logical_form = world.action_sequence_to_logical_form(action_sequence)
        denotation = world.execute(logical_form)
        return denotation if isinstance(denotation, list) else []

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. We only transform the action string sequences into logical
        forms here.
        """
        best_action_strings = output_dict["best_action_strings"]
        # Instantiating an empty world for getting logical forms.
        world = CSQALanguage(CSQAContext({}, {}, [], [], [], [], {}, {}))
        logical_forms = []
        for instance_action_sequences in best_action_strings:
            instance_logical_forms = []
            for action_strings in instance_action_sequences:
                if action_strings:
                    instance_logical_forms.append(world.action_sequence_to_logical_form(action_strings))
                else:
                    instance_logical_forms.append('')
            logical_forms.append(instance_logical_forms)

        action_mapping = output_dict['action_mapping']
        best_actions = output_dict['best_action_strings']
        debug_infos = output_dict['debug_info']
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions[0], debug_info):
                considered_actions = action_debug_info['considered_actions']
                probabilities = action_debug_info['probabilities']

                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append((action_mapping[(batch_index, action)], probability))
                actions.sort()
                considered_actions, probabilities = zip(*actions)

                action_info = {'predicted_action': predicted_action,
                               'considered_actions': considered_actions,
                               'action_probabilities': probabilities,
                               'question_attention': action_debug_info.get('question_attention', [])}

                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        output_dict["logical_form"] = logical_forms
        return output_dict

