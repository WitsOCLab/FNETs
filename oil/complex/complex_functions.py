from typing import Callable, Any, Optional
import torch
import torch.nn.functional as F
from torch import is_complex

def split_complex(z, fn:Callable[[Any],Any]):
    if is_complex(z):
        dtype = z.dtype
        return (fn(z.real)+1j*fn(z.imag)).type(dtype)
    else:
        return fn(z)
def split_complex_loss(z1, z2, criterion:Callable[[Any,Any],Any]):
    if is_complex(z):
        dtype = z.dtype
        return (criterion(z1.real, z2.real)+1j*criterion(z1.imag, z2.imag)).type(dtype)
    else:
        return callable(z)
    
# Complex functions aliases
def complex_clamp(z,min:Optional[torch.Tensor]=None,max:Optional[torch.Tensor]=None):
    if is_complex(z):
        dtype = z.dtype
        if min is not None and is_complex(torch.tensor(min)):
            min_r = min.real
            min_i = min.imag
        else:
            min_r = min
            min_i = min
        if max is not None and is_complex(torch.tensor(max)):
            max_r = max.real
            max_i = max.imag
        else:
            max_r = max
            max_i = max
        return (torch.clamp(z.real, min=min_r, max=max_r) + 1j*torch.clamp(z.imag, min=min_i, max=max_i)).type(dtype)
    else:
        return torch.clamp(z,min=min,max=max)
    
def complex_relu(z):
    return complex_clamp(z, min=0)
    
if __name__ == "__main__":
    import torch
    z = torch.tensor([1-2j], dtype=torch.complex64)
    fn = F.relu
    print(split_complex(z,fn))
    loss = torch.nn.MSELoss()
    print(split_complex_loss(z,z,loss))

    print(complex_relu(z))
    print(complex_relu(-z))
