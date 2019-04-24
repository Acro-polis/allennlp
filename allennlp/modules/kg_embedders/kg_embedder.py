import torch

from allennlp.common import Registrable
from allennlp.data import Vocabulary


class KgEmbedder(torch.nn.Module, Registrable):
    """
    A ``TokenEmbedder`` is a ``Module`` that takes as input a tensor with integer ids that have
    been output from a :class:`~allennlp.data.TokenIndexer` and outputs a vector per token in the
    input.  The input typically has shape ``(batch_size, num_tokens)`` or ``(batch_size,
    num_tokens, num_characters)``, and the output is of shape ``(batch_size, num_tokens,
    output_dim)``.  The simplest ``TokenEmbedder`` is just an embedding layer, but for
    character-level input, it could also be some kind of character encoder.

    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  This lets us
    more easily compute output dimensions for the :class:`~allennlp.modules.TextFieldEmbedder`,
    which we might need when defining model parameters such as LSTMs or linear layers, which need
    to know their input dimension before the layers are called.
    """
    default_implementation = "kg_embedding"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
        token.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def extend_vocab(self, extended_vocab: Vocabulary, vocab_namespace: str = None):
        """
        Assures that this ``TokenEmbedder`` can work with extended vocabulary.
        Unless overridden by concrete ``TokenEmbedder``, it is no-op.
        """
        pass