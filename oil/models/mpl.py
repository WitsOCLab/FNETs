"""
Contains an implementation of a basic multi-layer perception using pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Iterable

class MLP(nn.Module):
    def __init__(self, layers:int|Iterable[int]|None = None,
                 input_size:int|None = None,
                 hidden_size:int|Iterable[int]|None = None,
                 output_size:int|None = None,
                 n_hidden:int = 1,
                 activation_fn:Callable = F.relu,
                 activate_first:bool = True, 
                 activate_last:bool = True,
                 input_transform:Callable|None = None,
                 dtype:Any|None = None):
        super(MLP, self).__init__()

        if layers is None:
            layers = hidden_size
            if (hidden_size is None and n_hidden > 0) or layers is None:
                raise ValueError(f"If an MLP has hidden layers their size must be specified using the `layers` or `hidden_size` argument.")

        if isinstance(layers, (int, float)):
            # Layers are all same size
            extra_layers = sum([_size is None for _size in (input_size, output_size)])
            layers = [layers] * (n_hidden + 1 + extra_layers)
        elif isinstance(layers, Iterable) and not isinstance(layers, list):
            # Make layers a list to simplify operations
            layers = list(layers)

        # Set input and output layer
        if input_size is None:
            input_size = layers[0]
            layers = layers[1:]
            
        if output_size is not None:
            layers.append(output_size)

        self._in_size = input_size
        self._out_size = layers[-1]

        # Create a list of functions
        fcs = []
        prev_size = input_size
        for layer_size in layers:
            fcs.append(nn.Linear(prev_size, layer_size, dtype=dtype))
            prev_size = layer_size
        self.mod_list = nn.ModuleList(fcs)
        
        self.active_fn = activation_fn
        self.input_transform = input_transform

        self.activate_first = activate_first
        self.activate_last = activate_last


    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        for _i, module in enumerate(self.mod_list):
                if (_i == 0 and not self.activate_first) or (_i == len(self.mod_list)-1 and not self.activate_last):
                    x = module(x)
                else:
                    x = self.active_fn(module(x))

        return x
    
    def __repr__(self):
        return "MLP("+str(self.mod_list).strip("ModuleList(")
    
    @property
    def in_size(self)->int:
        return self._in_size
    @property
    def out_size(self)->int:
        return self._out_size
    @property
    def n_hidden(self)->int:
        return self.n_layers - 2
    @property
    def n_layers(self)->int:
        return len(self.mod_list)
    @property
    def layers(self):
        return self.mod_list


if __name__ == "__main__":
    mlp = MLP(layers=[1,2,3], dtype=torch.complex64)
    print(mlp)
