from typing import Callable

from torch import nn, Tensor

class WrappedModule(nn.Module):
    """
    A PyTorch nn.Module that wraps a callable loss function.
    """

    def __init__(self, fn:Callable[[Tensor, Tensor], Tensor]):
        """
        Initialise the `WrapperModule`.

        Args:
            fn (callable): Any Callable Function
        """
        super(WrappedModule, self).__init__()
        self.fn = fn

    def forward(self, *args)->Tensor:
        return self.fn(*args)
