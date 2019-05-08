import json

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('csqa-parser')
class CSQAParserPredictor(Predictor):
    pass
