
from torch.optim.lr_scheduler import _LRScheduler
from torch import optim
from typing import *

class TransformerScheduler(_LRScheduler):
    
    def __init__(self, optimizer: Union[optim.AdamW, optim.Adam], d_model = 512, lr_warmup_step = 100, **kwargs):
        
        self._optimizer = optimizer
        
        self._dmodel = d_model
        
        self._lr_warmup = lr_warmup_step

        # get the number of parameters
        self.len_param_groups = len(self._optimizer.param_groups)

        # provide the LRScheduler parameters
        super().__init__(self._optimizer, **kwargs)
        
    def get_lr(self):
        
        # recuperate the step number
        _step_num = self._step_count
        
        # calculate the learning rate
        lr = self._dmodel ** -0.5 * min(_step_num ** -0.5, 
                                              _step_num * self._lr_warmup ** -1.5)
        # provide the corresponding learning rate of each parameter vector
        # for updating
        return [lr] * self.len_param_groups

        
        
