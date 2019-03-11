from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric


@Metric.register("average_precision")
class AveragePrecision(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, relevant: List[str], retrieved: List[str]):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        relevant, retrieved = set(relevant), set(retrieved)
        if len(retrieved) != 0:
            instance_precision = len(relevant.intersection(retrieved)) / len(retrieved)
        else:
            instance_precision = 0
        self._total_value += instance_precision
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0

