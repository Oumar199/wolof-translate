"""This custom learning rate scheduler apply the learning rate scheduler with warmup according to the paper [Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case](https://arxiv.org/pdf/2001.08317)
"""

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class TransformerScheduler(_LRScheduler):
    def __init__(
        self, optimizer: Optimizer, d_model=100, lr_warmup_step=5000, **kwargs
    ):
        """Initialize the main attribute of the scheduler

        Args:
            optimizer (Optimizer): The optimizer
            d_model (int, optional): The embedding layer feature dimension. Defaults to 100.
            lr_warmup_step (int, optional): The number of warmup steps. Defaults to 5000.
        """

        self.d_model = d_model

        self.lr_warmup = lr_warmup_step

        self.len_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, **kwargs)

    def get_lr(self):
        """Get the new learning rates

        Returns:
            list: The learning rate of the different parameter groups
        """

        step_num = self.last_epoch + 1  # Increment step number from 1

        lr = self.d_model**-0.5 * min(
            step_num**-0.5, step_num * self.lr_warmup**-1.5
        )

        return [lr] * self.len_param_groups
