import torch
from typing import List

from overrides import overrides
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler


@MomentumScheduler.register("inverted_triangular")
class InvertedTriangular(MomentumScheduler):
    """
    Adjust momentum during training according to an inverted triangle-like schedule.

    The momentum starts off high, then decreases linearly for ``cool_down`` epochs,
    until reaching ``1 / ratio`` th of the original value. Then the momentum increases
    linearly for ``warm_up`` epochs until reaching its original value again. If there
    are still more epochs left over to train, the momentum will stay flat at the original
    value.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cool_down: int,
                 warm_up: int,
                 num_steps_per_epoch: int,
                 ratio: int = 10,
                 last_epoch: int = -1) -> None:
        self.cool_down = cool_down
        self.warm_up = warm_up
        self.ratio = ratio
        self.batch_num_total_epoch_end: List[int] = []
        self.num_steps_per_epoch = num_steps_per_epoch
        self.last_batch_num_total = -1
        super().__init__(optimizer, last_epoch)
        self.step_batch(0)

    def step_batch(self, batch_num_total: int = None):
        if batch_num_total is None:
            batch_num_total = self.last_batch_num_total + 1
        self.last_batch_num_total = batch_num_total
        for param_group, momentum in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = momentum

    def get_values(self):
        # step = self.last_epoch + 1
        step = self.last_batch_num_total / self.num_steps_per_epoch
        base_values = self.base_values

        # unpack Adam params
        if not isinstance(self.optimizer, torch.optim.SGD):
            base_values, b2 = zip(*self.base_values)

        if step <= self.cool_down:
            values = [m  - (m - m / self.ratio) * (step / self.cool_down)
                      for m in base_values]
        elif step <= self.cool_down + self.warm_up:
            values = [(m / self.ratio) + (m - m / self.ratio) * (step - self.cool_down) / self.warm_up
                      for m in base_values]
        else:
            values = base_values

        # repack adam params
        if not isinstance(self.optimizer, torch.optim.SGD):
            values = list(zip(values, b2))

        return values
