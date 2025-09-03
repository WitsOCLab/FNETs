from typing import Iterable, Callable, Any
from torch import Tensor
from torch.nn import Module

class CombinedLoss(Module):
    def __init__(self, loss_functions:Iterable[Callable[[Any,Any],Tensor|float]],
                 weights:Iterable[float|int|Tensor]|None = None,
                 normalize:bool = True):
        super(CombinedLoss, self).__init__()

        self._loss_functions = loss_functions

        if weights is None:
            weights = [1.0 for _ in loss_functions]
        self._weights = weights

        if normalize:
            weights_sum = sum(self._weights)
            self._weights = [_w/weights_sum for _w in self._weights]

    def forward(self,x1,x2):
        loss = 0.0
        for w, loss_fn in zip(self._weights, self._loss_functions):
            loss += w * loss_fn(x1,x2)
        return loss
    
    def __repr__(self):
        s = super().__repr__().strip(')')
        for w, loss_fn in zip(self._weights, self._loss_functions):
            s += f"\n\t{str(w)}*{str(loss_fn)}"
        s += "\n)"
        return s
