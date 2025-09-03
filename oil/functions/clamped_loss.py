from typing import Optional, Callable, Any, Literal
import numpy as np
import torch
from torch import Tensor, nn

class ClampedLoss(nn.Module):
    """
    A loss function that only applies if the output falls within a certain region.
    """

    def __init__(self,
                 min:Optional[torch.NumberType|np.number|Any] = None,
                 max:Optional[torch.NumberType|np.number|Any] = None,
                 criterion:Optional[Callable[[Tensor, Tensor], Tensor]] = nn.functional.l1_loss,
                 compare_to:Literal['bound', 'target'] = 'bound', # What do we compare x to when finding loss, bound exceeded (min/max) or target value
                 loss_region:Literal['between', 'outside'] = 'outside',
                 multiplier:torch.NumberType|np.number|Any = 1.):
        super(ClampedLoss, self).__init__()

        self.min = min
        self.max = max
        self.criterion = criterion
        self.compare_to = compare_to.lower()
        self.loss_region = loss_region.lower()
        self.multiplier = multiplier

        if self.min is None and self.max is None:
            raise Warning("ClampedLoss object created with no clamp values")
        if self.min is not None and self.max is not None and self.min > self.max: # type: ignore
            raise ValueError(f"Cannot clamp with a min of {min} and a max of {max}. Requires min <= max.")
        
        if self.criterion is None:
            self.criterion = lambda x,_: x

    def forward(self, x:Tensor, target:Tensor):

        # Find min region
        if self.min is None:
            min_comp = torch.ones_like(x)
        elif self.loss_region == 'outside':
            min_comp = x < self.min # type: ignore
        else:
            min_comp = x > self.min # type: ignore
        if self.compare_to == 'target':
            min_val = target
        else:
            min_val = self.min * torch.ones_like(x) # type: ignore


          # Find max region
        if self.max is None:
            max_comp = torch.ones_like(x)
        elif self.loss_region == 'outside':
            max_comp = x > self.max # type: ignore
        else:
            max_comp = x < self.min # type: ignore
        if self.compare_to == 'target':
            max_val = target
        else:
            max_val = self.min * torch.ones_like(x) # type: ignore

        return self.criterion(x*min_comp, min_val*min_comp) + self.criterion(x*max_comp, max_val*max_comp) # type: ignore

    def __repr__(self):
        if self.min is None and self.max is None:
            return self.criterion.__repr__()
        try:
            _str = f"{self.criterion.__repr__()} for "
        except TypeError:
            _str = "CondtionalLoss for: "
        if self.loss_region == 'outside':
            if self.min is not None:
                _str += f"x<{self.min}"
            if self.min is not None and self.max is not None:
                _str += " U "
            if self.max is not None:
                _str += f"x>{self.max}"
        else:
            if self.min is not None and self.max is None:
                _str += f"x>{self.min}"
            elif self.min is not None and self.max is not None:
                _str += f"{self.min}<x<{self.max}"
            elif self.max is not None:
                _str += f"x>{self.max}"

        return _str




if __name__ == "__main__":
    a= ClampedLoss(0,1)
    print(a)
    t1 = torch.tensor([3., 0.5, -1])
    t2 = torch.tensor([3., 0.5, -1])
    ans = a(t1,t2)
    print(ans)
