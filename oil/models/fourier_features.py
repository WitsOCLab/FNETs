"""
A set of positionally encoded fourier features for training high-frequency models
"""
from typing import Optional, Callable, Iterable, Collection, Type
from enum import Enum

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from oil.models.mpl import MLP

def geometric_generator_base_2(base_frequency:float, n_features:int)->Tensor:
    return torch.pow(2.0, torch.arange(n_features)) * base_frequency

class FourierFeatures(nn.Module):
    def __init__(self, n_inputs:int,
                 n_features:int = 8, base_frequency:float|Iterable = 1.0,
                 frequency_generator:Callable[[float, int],Tensor] = geometric_generator_base_2,
                 positional_transforms:Iterable[Optional[Callable[[Tensor], Tensor]]] =  [None],
                 dtype:Optional[torch.dtype] = None
                 ) -> None:
        super(FourierFeatures, self).__init__()

        if isinstance(base_frequency, Iterable):
            scales = torch.tensor(base_frequency,dtype=dtype)
            base_frequency = scales[0]

            n_features = len(scales)
        else:
            scales = frequency_generator(base_frequency, n_features).to(dtype)
        self.frequencies = nn.Parameter(scales, requires_grad=False)

        self._positional_transforms = list(positional_transforms)

        self._in_size = n_inputs
        self._out_size = n_inputs + (n_features * 2 + 1) * (len(self._positional_transforms))


    def forward(self, x:torch.Tensor):
        if self._in_size != 0:
            d_norm = x[...,-1]
            _out = x[...,:-1]
            while x.dim() > _out.dim():
                _out = _out.unsqueeze(0)
        else:
            d_norm = x
            _out = None

        
        for transform in self._positional_transforms:
            if transform is None:
                val = d_norm
            else:
                val = transform(d_norm)
            periodic_args = val * self.frequencies * 2 * torch.pi

            while x.dim() > periodic_args.dim():
                periodic_args = periodic_args.unsqueeze(0)
            while x.dim() > val.dim():
                val = val.unsqueeze(0)

            sin_features = torch.sin(periodic_args)
            cos_features = torch.cos(periodic_args)

            if _out is None:
                _out = torch.cat([
                    val,
                    sin_features,
                    cos_features
                ], dim=-1)
            else:
                _out = torch.cat([
                    _out,
                    val,
                    sin_features,
                    cos_features
                ], dim=-1)
    
        return _out
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, frequencies={self.n_frequencies}"
    
    
    @property
    def in_size(self)->int:
        return self._in_size + 1
    @property
    def in_features(self)->int:
        return self.in_size
    @property
    def out_size(self)->int:
        return self._out_size
    @property
    def out_features(self)->int:
        return self.out_size    
    @property
    def positional_transforms(self):
        return self._positional_transforms
    @property
    def radial_frequencies(self):
        return 2*torch.pi*self.frequencies
    @property
    def n_frequency_features(self):
        return 2*self.n_frequencies
    @property
    def n_frequencies(self):
        return len(self.frequencies)      
    @property
    def angular_frequencies(self):
        return 2*torch.pi*self.frequencies
    

    @staticmethod
    def geometric_base_2_generator(base_frequency:float, n_features:int)->Tensor:
        return torch.pow(2.0, torch.arange(n_features)) * base_frequency

class FourierFeatureMLP(nn.Module):
    def __init__(self, n_inputs:int, hidden_layers:Iterable[int]=[20], n_outputs:Optional[int]=None,
                 n_features:int = 8, base_frequency:float|Iterable|Collection = 1.0,
                 frequency_generator:Callable[[float, int],Tensor] = FourierFeatures.geometric_base_2_generator,
                 positional_transforms:Iterable[Optional[Callable[[Tensor], Tensor]]] =  [None],
                 activation_fn:Callable[[Tensor], Tensor] = F.hardtanh, activate_first:bool = True, activate_last:bool = False,
                 dtype:torch.dtype = torch.float32) -> None:
        super(FourierFeatureMLP,self).__init__()

        self.features = FourierFeatures(
            n_inputs=n_inputs,
            n_features=n_features,
            base_frequency=base_frequency,
            frequency_generator=frequency_generator,
            positional_transforms=positional_transforms,
            dtype=dtype
        )
        
        _in_size = n_inputs + (self.features.n_frequency_features + 1) * (len(self.features.positional_transforms))
        self._in_size = n_inputs

        if n_outputs is None:
            n_outputs = n_inputs
        _out_size = n_outputs

        self.mlp = MLP(layers=[_in_size, *hidden_layers, _out_size],
                       activation_fn=activation_fn,
                       activate_first=activate_first,
                       activate_last=activate_last,
                       dtype=dtype)
        
    def forward(self, x:torch.Tensor):
        x = self.features(x)
        return self.mlp(x)   
   
    @property
    def in_size(self)->int:
        return self._in_size + 1
    @property
    def out_size(self)->int:
        return self.mlp.out_size
    @property
    def n_hidden(self)->int:
        return self.n_layers - 2
    @property
    def n_layers(self)->int:
        return self.mlp.n_layers
    
    @property
    def frequencies(self):
        return self.features.frequencies
    
    @property
    def angular_frequencies(self):
        return self.features.ang
    
    @property
    def frequency_weights(self):
        """
        Returns the the weights of each of the frequency features
        """
        w = self.mlp.mod_list[0].state_dict()['weight']
        f_w = torch.sum(torch.abs(w), dim=0)[1:]
        f_w = torch.sqrt((f_w[:len(f_w)//2])**2 + (f_w[len(f_w)//2:])**2)

        return f_w

        
if __name__ == "__main__":
    a = FourierFeatureMLP(4, base_frequency=[1,2,3,4,5])
    print(a)
    print(a.features.frequencies)
    print(a.features.radial_frequencies)

