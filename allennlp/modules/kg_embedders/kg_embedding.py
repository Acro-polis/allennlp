import io
import tarfile
import zipfile
import re
import logging
import warnings
import itertools
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple, Dict

from overrides import overrides
import numpy as np
import torch
from torch.nn.functional import embedding

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path
from allennlp.data import Vocabulary
from allennlp.modules.kg_embedders.kg_embedder import KgEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@KgEmbedder.register("kg_embedding")
class KgEmbedding(KgEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_params``

    Note that if you are using our data API and are trying to embed a
    :class:`~allennlp.data.fields.TextField`, you should use a
    :class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.

    Parameters
    ----------
    num_embeddings : int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    projection_dim : int, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False):
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False):
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : str, (optional, default=None):
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.

    Returns
    -------
    An Embedding module.
    """
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 entity_weight: torch.FloatTensor = None,
                 relation_weight: torch.FloatTensor = None,
                 entity2id: Dict[str, int] = None,
                 relation2id: Dict[str, int] = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 vocab_namespace: str = None) -> None:
        super(KgEmbedding, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.padding_index = padding_index
        self.trainable = trainable
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace

        self.output_dim = projection_dim or embedding_dim

        # TODO refactor
        if entity_weight is None:
            entity_weight = torch.FloatTensor(num_entities, embedding_dim)
            self.entity_weight = torch.nn.Parameter(entity_weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.entity_weight)
        else:
            if entity_weight.size() != (num_entities, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.entity_weight = torch.nn.Parameter(entity_weight, requires_grad=trainable)

        if relation_weight is None:
            relation_weight = torch.FloatTensor(num_relations, embedding_dim)
            self.relation_weight = torch.nn.Parameter(relation_weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.relation_weight)
        else:
            if relation_weight.size() != (num_relations, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.relation_weight = torch.nn.Parameter(relation_weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.entity_weight.data[self.padding_index].fill_(0)
            self.relation_weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection_entity = torch.nn.Linear(embedding_dim, projection_dim)
            self._projection_relation = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection_entity = None
            self._projection_relation = None

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs, input_type='entity'):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.

        # TODO
        # Map entities to ids of embedding matrix and pad with zeros.
        if input_type == 'entity':
            weight = self.entity_weight
            mapping2id = self.entity2id
        elif input_type == 'relation':
            weight = self.relation_weight
            mapping2id = self.relation2id

        max_len = max([len(input) for input in inputs])
        for i, input in enumerate(inputs):
            ids = []
            for key in input:
                if key not in mapping2id:
                    mapping2id[key] = len(weight)
                    new_embedding = torch.FloatTensor(1, self.embedding_dim)
                    new_embedding = torch.nn.Parameter(new_embedding)
                    torch.nn.init.xavier_uniform_(new_embedding)
                    weight = torch.cat((weight, new_embedding), 0)
                    if input_type == 'entity':
                        self.entity_weight = torch.nn.Parameter(weight, requires_grad=self.trainable)
                    elif input_type == 'relation':
                        self.relation_weight = torch.nn.Parameter(weight, requires_grad=self.trainable)
                ids.append(mapping2id[key])
            inputs[i] = ids + [0] * (max_len - len(input))
        inputs = torch.LongTensor(inputs)

        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        embedded = embedding(inputs, weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if input_type == 'entity':
            if self._projection_entity:
                projection = self._projection_entity
                for _ in range(embedded.dim() - 2):
                    projection = TimeDistributed(projection)
                embedded = projection(embedded)
        elif input_type == 'relation':
            if self._projection_relation:
                projection = self._projection_relation
                for _ in range(embedded.dim() - 2):
                    projection = TimeDistributed(projection)
                embedded = projection(embedded)

        return embedded

    @classmethod
    def from_params(cls, params: Params) -> 'Embedding':
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.

        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file or an URL of a (cached) remote file.
        """
        # pylint: disable=arguments-differ
        num_entities = params.pop_int('num_entities', None)
        num_relations = params.pop_int('num_relations', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        embedding_dim = params.pop_int('embedding_dim')
        entity_pretrained_file = params.pop("entity_pretrained_file", None)
        relation_pretrained_file = params.pop("relation_pretrained_file", None)
        entity2id_file = params.pop('entity2id_file', None)
        relation2id_file = params.pop('relation2id_file', None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        params.assert_empty(cls.__name__)

        entity2id = {}
        with open(entity2id_file, 'r') as f:
            next(f)
            for line in f:
                key_value = line.split('\t')
                entity2id[key_value[0]] = int(key_value[1])
        entity_vocab_size = len(entity2id)

        relation2id = {}
        with open(relation2id_file, 'r') as f:
            next(f)
            for line in f:
                key_value = line.split('\t')
                relation2id[key_value[0]] = int(key_value[1])
        relation_vocab_size = len(relation2id)

        if entity_pretrained_file:
            entity_weight = _read_pretrained_embeddings_file(entity_pretrained_file,
                                                             embedding_dim,
                                                             entity_vocab_size)
        else:
            entity_weight = None

        if relation_pretrained_file:
            relation_weight = _read_pretrained_embeddings_file(relation_pretrained_file,
                                                               embedding_dim,
                                                               relation_vocab_size)
        else:
            relation_weight = None

        return cls(num_entities=num_entities,
                   num_relations=num_relations,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   entity2id=entity2id,
                   relation2id=relation2id,
                   entity_weight=entity_weight,
                   relation_weight=relation_weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   vocab_namespace=vocab_namespace)


def _read_pretrained_embeddings_file(path: str,
                                     embedding_dim: int,
                                     vocab_size: int) -> torch.FloatTensor:
    embeddings = np.memmap(path, dtype='float32', mode='r', shape=(vocab_size, embedding_dim))
    return torch.FloatTensor(embeddings)
