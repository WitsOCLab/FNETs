from typing import Iterable, Callable, Optional
from torch import Tensor, tensor, NumberType
from torch.nn import Module
from abc import ABC, abstractmethod

class EvolvingLoss(Module,ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    @abstractmethod
    def update(self, epoch:Optional[float|int]):
        pass

class EvolvingWeightLoss(EvolvingLoss):
    """
    An error function that can be updated during training.
    The `update` function should be called at the start of each training epoch
    """
    def __init__(self,
                 criterion:Callable[[Tensor,Tensor],Tensor],
                 weight_function:Optional[Callable[[float|Tensor],float]] = None,
                 initial_weight:Optional[float|Tensor] = None,
                 max_weight:Optional[float|Tensor] = None,
                 use_internal_counter:bool = False,
                 n_epochs:Optional[int] = None,
                 static:bool = False
                 ) -> None:
        super().__init__()

        self.criterion = criterion
        self.weight_fn = weight_function
        self.counter = 0
        self.use_counter = use_internal_counter
        self.n_epochs = n_epochs
        self.max_weight = max_weight
        self.is_static = static


        if initial_weight is None:
            initial_weight = self._get_weight(0)
        self.weight = initial_weight

    def forward(self,x:Tensor,target:Tensor):
        return self.weight * self.criterion(x,target)
    
    def update(self, epoch:Optional[float|int]):
        if self.is_static:
            return
        _normalize =  self.n_epochs is not None and self.n_epochs > 0
        if self.use_counter or epoch is None:
            self.counter += 1
            val = self.counter
        else:
            val = epoch
            _normalize = _normalize and not isinstance(epoch,float)

        if _normalize and self.n_epochs is not None:
            val /= self.n_epochs 

        self.weight = self._get_weight(val)


    def _get_weight(self,val:float|Tensor):
        if self.weight_fn is None:
            return val
        return self.weight_fn(val)

class CombinedEvolvingWeightLoss(EvolvingLoss):
    def __init__(self, evolving_errors:Iterable[EvolvingWeightLoss], normalize:bool = True) -> None:
        super().__init__()

        self.criteria = evolving_errors
        self.normalize = normalize

    def forward(self,x:Tensor,target:Tensor):
        loss = 0
        for e in self.criteria:
            loss += e(x,target)
        
        if self.normalize and loss != 0:
            combined_weights = combined_weights = self._get_combined_weights()
            loss /= combined_weights
        return loss
        
            
    def update(self, epoch: float | int | None):
        for e in self.criteria:
            e.update(epoch)

    @property
    def weighs(self):
        
        weights = []
        combined_weights = combined_weights = self._get_combined_weights()
        for e in self.criteria:
            weights.append(e.weight/combined_weights)

        return weights
    
    def _get_combined_weights(self):
        if self.normalize:
            combined_weights = 0.
            for e in self.criteria:
                combined_weights += float(e.weight)
            if combined_weights != 0:
                return combined_weights
        return 1.

        
if __name__ == "__main__":
    from torch import nn

    b = EvolvingWeightLoss(criterion=lambda x,t:2., n_epochs=10) # type: ignore #
    c = EvolvingWeightLoss(criterion=lambda x,t:3.,weight_function=lambda x:2-2*x, n_epochs=10) # type: ignore #
    a = CombinedEvolvingWeightLoss(
        [b,c],
        normalize=True
    )
    print(a.criteria)
    for e in range(10):
        a.update(e)

        print(e/10)


        print(sum(a.weighs))
        print(a.weighs)
        print(a(1,1))

        