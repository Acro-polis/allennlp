# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.average_recall import AverageRecall


class F1MeasureTest(AllenNlpTestCase):
    @staticmethod
    def test_average_recall():
        avg_recall = AverageRecall()

        relevant = [str(i) for i in range(10, 20)]
        retrieved = [str(i) for i in range(20)]
        avg_recall(relevant, retrieved)
        assert avg_recall.get_metric(reset=True) == 1.0

        relevant = [str(i) for i in range(20)]
        retrieved = [str(i) for i in range(10, 30)]
        avg_recall(relevant, retrieved)
        assert avg_recall.get_metric() == 0.5

        relevant = [str(i) for i in range(10, 20)]
        retrieved = [str(i) for i in range(10, 20)]
        avg_recall(relevant, retrieved)
        assert avg_recall.get_metric() == 0.75

        avg_recall.reset()
        assert avg_recall.get_metric() == 0.
