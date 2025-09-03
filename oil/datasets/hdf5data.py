"""
Classes and methods used to handle hdf5 datasets files with pytorch
"""
from typing import Callable, Iterable, Literal, List
import h5py, os
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy, tensor, float32

LEN_KEY = "length"

class HDF5Dataset(Dataset):
    """
    Alternative HDF5Dataset that can have store multiple data, not just key and values pairs
    """
    SETS_KEY = "subsets"
    SET_PREFIX = "subset_"
    def __init__(self, path:str,
                 subsets:Iterable[str]|int = ["data", "label"],
                 overwrite:bool = False, data_transform:Callable|None = None,
                 as_tensor:bool = True,
                 as_tuple:bool|None = None):
        super(HDF5Dataset, self).__init__()

        self._path = path
        self.f = None
        self.is_open = False
        self.data_transform = data_transform
        self.as_tensor = as_tensor
        self._active_subsets = []


        # make keys for existing datasets
        if isinstance(subsets, int):
            # Make generic subsets
            _tmp_subsets = []
            for _i in range(subsets):
                _set_name = f"{self.SET_PREFIX}{_i}"
                _tmp_subsets.append(_set_name)
            subsets = _tmp_subsets
        elif not isinstance(subsets, List):
            subsets = list(subsets)


        # Remove existing file if overwrite is required
        if overwrite and os.path.isfile(path):
            self.close()
            os.remove(path)
        if os.path.isfile(path):
            with h5py.File(self._path, mode='r') as f:
                _subsets = f[self.SETS_KEY]
                assert isinstance(_subsets, h5py.Dataset)
                # Retrieve existing subsets 
                self._subsets = [str(subset, 'utf-8') for subset in _subsets[()]] 
                self._active_subsets = [subset for subset in subsets if subset in self._subsets]
        else:
            if os.path.dirname(path) != '':
                os.makedirs(os.path.dirname(path), exist_ok=True)
            

            self._subsets = list(subsets)
                

            with h5py.File(self._path, mode='w') as f:
                f.create_dataset(LEN_KEY, data=0)
                f.create_dataset(self.SETS_KEY, data=self._subsets)

        # If there are two subsets (say key and value) the output data as a tuple
        if as_tuple is None:
            as_tuple = (self.subset_count == 2)
        self.as_tuple = as_tuple

                

    
    def __del__(self):
        self.close()
       
    def __len__(self):
        return self.length
    
    def __getitem__(self, index)->dict|tuple:
        # Adjust for negative index
        if index < 0:
            index = len(self) + index # Not index is negative
        _subsets = self.active_subsets
        _item = {}
        f =self.get_f('r')
        for subset in _subsets:
            _data = f[f"{subset}{index}"]
            assert isinstance(_data,h5py.Dataset)
            data = _data[()]
            if self.as_tensor:
                if isinstance(data, np.ndarray):
                    data = from_numpy(data)
                elif isinstance(data, (int, float, complex, np.number)):
                    data = tensor(data)
            _item[subset] = data

        if self.data_transform is not None:
            _item = self.data_transform(_item)

        if self.as_tuple and not isinstance(_item, tuple):
            return tuple(_item.values())

        return _item
    
    def __enter__(self):
        if not self.is_open:
            self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.is_open:
            self.close()
        return False
    
    def open(self, mode:str='r+')->"HDF5Dataset":
        if not self.is_open or self.f is None:
            self.f = h5py.File(self._path, mode=mode)
            self.is_open =True
        
        return self
    
    def close(self):
        if self.is_open:
            if self.f is not None:
                self.f.close()
                self.f = None
            self.is_open = False
        elif self.f is not None:
            self.f.close()
        

    
    def add_data(self, data:Iterable|dict):
        f = self.get_f('r+')
        _len = self.length
        if isinstance(data, dict):
            for _subset in self._subsets:
                f.create_dataset(f"{_subset}{_len}", data=data[_subset])
        else:
            for _data, _subset in zip(data, self._subsets):
                f.create_dataset(f"{_subset}{_len}", data=_data)

        _len += 1
        len_set = f[LEN_KEY]
        assert(isinstance(len_set,h5py.Dataset))
        len_set[...] = _len
    def add(self, data:Iterable|dict):
        self.add_data(data)

    def get_f(self, mode:str ='r+')->h5py.File:
        if not self.is_open or self.f is None:
            self.open(mode)
        elif mode != self.f.mode:
            self.close()
            self.open(mode)
        assert self.f is not None
        return self.f
    
    def __repr__(self):
        _string = "HDFDataset(\n"
        _string += f"  File: {self.path}\n  Length: {len(self)}\n  Subsets: {self.subset_names}\n)"
        return _string
    

    
    @property
    def subset_names(self)->List[str]:
        return self._subsets
    @property
    def subsets(self)->List[str]:
        return self._subsets
    
    @property
    def subset_count(self)->int:
        return len(self._subsets)
    @property
    def n_subsets(self)->int:
        return len(self._subsets)
    
    @property
    def path(self):
        return self._path
    
    @property
    def length(self):
        f = self.get_f('r')
        return int(f[LEN_KEY][()]) # type: ignore

    @classmethod
    def load(cls, path:str, data_transform:Callable|None = None)->"HDF5Dataset":
        if not os.path.isfile(path):
            raise FileNotFoundError()
        return HDF5Dataset(path, overwrite=False, data_transform=data_transform)
    

    # Control limited subsets
    def set_active_subsets(self, subsets:Iterable[str]|Literal['all']):
        if subsets == 'all':
            self._active_subsets = []
        else:
            self._active_subsets = list(subsets)

    @property
    def active_subsets(self)->List[str]:
        if self._active_subsets is not None and len(self._active_subsets) > 0:
            return self._active_subsets
        return self.subsets

    