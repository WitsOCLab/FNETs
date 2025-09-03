from typing import Callable
import torch
from torch import Tensor, tensor, nn

class ConditionalLoss(nn.Module):
    """
     A callable that applies a loss function only when a given condtion is true.
    """
    def __init__(self,
                 condition:Callable[[Tensor, Tensor],Tensor],
                 criterion:Callable[[Tensor, Tensor],Tensor] = nn.functional.l1_loss):
        """
        Initialise the ConditionalLoss.
        """
        super(ConditionalLoss, self).__init__()
        self.condtion = condition
        self.criterion = criterion

    def forward(self, x:Tensor, target:Tensor):
        _condtional_filter = self.condtion(x,target)
        return self.criterion(x*_condtional_filter, target*_condtional_filter)
    
    def __repr__(self):
        return f"Conditional {self.criterion.__repr__()}"
    
