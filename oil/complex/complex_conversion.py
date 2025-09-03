"""
A Collection of functions and classes used to convert between float and complex tensors.
Doing this explicitly may be more efficient than using complex tensor wrappers.
"""
import torch
from typing import Literal, Optional, Any, Collection, Tuple, Sequence
from enum import Enum, unique
from torch import Tensor, is_complex    

class ExpandMethod(Enum):
    """
    Describes how a tensor should be extended to incorporate the two real values required to represent a complex value.
    """
    CONCAT = 'concat'
    "Concatenate the now complex values into a single tensor without changing dimensions"
    STACK = 'stack'
    "Stack the two-valued complex representation to occupy seperate dimentions"
    ALTERNATE = 'alternate'

class ComplexRepresentation(Enum):
    REAL_IMAG = 'real-imag'
    MOD_ARG = 'mod-arg'
    MOD_SIN_COS = 'mod-sin-cos'

    @property
    def n_vals(self):
        if self in [ComplexRepresentation.MOD_SIN_COS]:
            return 3
        else:
            return 2

def complex_to_float(z:Tensor,
                     expand_method:Literal['concat', 'stack', 'alternate']|ExpandMethod='stack',
                     representation:Literal['real-imag', 'mod-arg']|ComplexRepresentation='real-imag',
                     arg_range:Tuple[float,float]|Sequence[float] = (0, 1),
                     normalize_arg:bool = True,
                     dim:int = -1)->Tensor:
    """
    Converts a complex tensor 
    """

    expand_method = ExpandMethod(expand_method)

    try:
        representation = ComplexRepresentation(representation)
    except:
        raise ValueError(f"Complex representation'{representation}' is not supported")


    if 'real-imag' == str.lower(representation.value):
        val1 = z.real
        try:
            val2 = z.imag
        except RuntimeError:
            val2 = torch.zeros_like(val1)

        vals = (val1, val2)
    elif 'mod-arg' == str.lower(representation.value):
        val1 = z.abs()
        val2 = z.angle()
        if normalize_arg:
            val2 = (val2*0.5)/torch.pi
            _min_pt = -0.5
            if arg_range[0] >= 0:
                val2 = val2%1
                _min_pt = 0
            if arg_range[0] != 0 or arg_range[1] != 1:
                # Scale arg to correct range
                _scale_fact = arg_range[1]-arg_range[0]
                
                val2=(val2-_min_pt)*_scale_fact+arg_range[0]

        vals = (val1, val2)
    elif representation == ComplexRepresentation.MOD_SIN_COS:
        val1 = z.abs()
        a = z.angle()
        val2 = torch.sin(a)
        val3 = torch.cos(a)
        vals = (val1,val2,val3)
    else:
        raise ValueError(f"Complex representation'{representation}' is not supported")


    expand_method = ExpandMethod(expand_method)
    
    if expand_method == ExpandMethod.CONCAT:
        return torch.cat(vals, dim)
    elif expand_method == ExpandMethod.STACK:
        return torch.stack(vals, dim)
    elif expand_method == ExpandMethod.ALTERNATE:
        _dim = z.ndim
        _shape = [_s for _s in z.shape]
        if len(_shape) == 0:
            return complex_to_float(z, 'stack', normalize_arg=normalize_arg, arg_range=arg_range, dim=dim)
        _shape[-(1+dim)]*=len(vals)
        return torch.stack(vals, dim=_dim).reshape(_shape)
    else:
        raise ValueError(f"'{expand_method}' is not a valid complex_to_float expand method.")
    
def float_to_complex(x:Tensor,
                     expand_method:Literal['concat', 'stack', 'alternate']|ExpandMethod='stack',
                     representation:Literal['real-imag', 'mod-arg']|ComplexRepresentation='real-imag',
                     arg_range:Tuple[float,float]|Sequence[float] = (0, 1),
                     normalize_arg:bool = True,
                     dim:int = -1)->Tensor:
    
    expand_method = ExpandMethod(expand_method)
    representation = ComplexRepresentation(representation)

    if expand_method == ExpandMethod.CONCAT:
        _shape = x.shape[dim]
        n_vals = representation.n_vals
        vals = torch.split(x, _shape//n_vals, dim=dim)
    elif expand_method == ExpandMethod.STACK:
        vals = torch.unbind(x, dim=dim)
    elif expand_method == ExpandMethod.ALTERNATE:
        n_vals = representation.n_vals
        _shape = [s for s in x.shape]
        _shape[-(1+dim)]//=n_vals
        vals = []
        for _i in range(n_vals):
            vals.append(x.flatten()[_i::n_vals].reshape(_shape))
    else:
        raise ValueError(f"'{expand_method}' is not a valid complex_to_float expand method.")
    
    if representation == ComplexRepresentation.REAL_IMAG:
        return vals[0] + 1j * vals[1]
    elif representation == ComplexRepresentation.MOD_ARG:
        _val1 = vals[0]
        _val2 = vals[1]
        if normalize_arg:
            _val2=_val2*2*torch.pi # Do not use inplace transform to ensure that original remains unchanged.
            if arg_range[0] != 0 or arg_range[1] != 1:
                # Reverse value shift
                if arg_range[0] < 0:
                    _min_pt = -0.5
                else:
                    _min_pt = 0
                _scale_fact = arg_range[1]-arg_range[0]
                _val2 = (_val2-arg_range[0])/_scale_fact+_min_pt
                
        return _val1 * (torch.cos(_val2)+1j*torch.sin(_val2))
    elif representation == ComplexRepresentation.MOD_SIN_COS:
        return vals[0] * (vals[2] + 1j * vals[1])
    else:
        raise ValueError(f"'{representation} is not a valid complex reorientation.")
    

# def get_mod(x:Tensor,
#                      expand_method:Literal['concat', 'stack', 'alternate']='stack',
#                      representation:Literal['real-imag', 'mod-arg']='real-imag',
#                      arg_range:Tuple[float,float]|Collection[float] = (0, 1),
#                      normalize_arg:bool = True,
#                      dim:int = 0)->Tensor:
    
    
class ComplexConverter(object):
    """
    A simple object used for multiple complex conversions to and from the same pseudo-complex representation.
    """
    def __init__(self,
                 expand_method:Literal['concat', 'stack', 'alternate']|str='stack',
                 representation:Literal['real-imag', 'mod-arg']|str='real-imag',
                 arg_range:Tuple[float,float]|Sequence[float] = (0, 1),
                 normalize_arg:bool = True):
        self.expand_method = ExpandMethod(expand_method)
        self.representation = ComplexRepresentation(representation)
        self.arg_range = arg_range
        self.normalize_arg = normalize_arg

    def to_complex(self, x:Tensor):
        return float_to_complex(x, self.expand_method, self.representation, self.arg_range, self.normalize_arg)
    def to_float(self, z:Tensor):
        return complex_to_float(z, self.expand_method, self.representation, self.arg_range, self.normalize_arg)
    # def get_abs(self, z:Tensor):
    #     """Returns the absolute value of the complex number"""

    
    def __call__(self, x):
        if is_complex(x):
            return self.to_float(x)
        else:
            return self.to_complex(x)
        
    

if __name__ == "__main__":
    b = ComplexRepresentation.MOD_SIN_COS
    print(b.n_vals)
    #a = torch.tensor([[1,2j,3+4j],[-4j,-5,1+6j]], dtype=torch.complex64).unsqueeze(0)

    a = torch.tensor([1,2j], dtype=torch.complex64).unsqueeze(0).unsqueeze(0)

    cc = ComplexConverter('concat','mod-sin-cos')
    b = cc.to_float(a)

    print('a:\n',a)
    print("b:\n",b)
    c = cc.to_complex(b)
    # b = complex_to_float(a, representation='mod-arg', arg_range=[-0.5,0.5])
    #c = float_to_complex(b, representation='mod-arg', arg_range=[-0.5,0.5])
    #c = float_to_complex(b, representation='real-imag', dim=1)

    print("c:\n",c)
    

    exit()
    s = complex_to_float(a,'stack')
    c = complex_to_float(a,'concat')
    al = complex_to_float(a,'alternate', 'mod-arg', arg_range=[0,1])
    print("stack ", s)
    print("Cat ",c)
    print("Alternate ", al)

    _s = c.shape[0]
    print(torch.split(c,_s//2))

    a1 = float_to_complex(c, 'concat')
    a2 = float_to_complex(s,'stack')
    a3 = float_to_complex(al,'alternate', 'mod-arg', arg_range=[0,1])
    print(a1)
    print(a2)
    print(a3)
    print("------------")
    print(al)

    z = torch.tensor(2+3j)
    x = complex_to_float(z,'alternate')
    print(x)