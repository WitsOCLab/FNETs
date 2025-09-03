# """
# Loss functon for SSIM 
# """
# from typing import List, Tuple, Sequence
# from torch import Tensor, nn
# from torch.nn import Module
# from pytorch_msssim import SSIM

# class SSIMLoss(Module):
#     def __init__(self, data_range:float|int=255,
#                  size_average:bool = True,
#                  window_size:int = 11,
#                  window_sigma:float = 1.5,
#                  n_channels:int = 3,
#                  spatial_dims:int = 2,
#                  k:Tuple[float, float]|List[float]|Sequence[float] = (0.01, 0.03),
#                  nonnegative_ssim:bool = True,
#                  is_loss:bool = True,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._is_loss = is_loss
#         self._input_dims = spatial_dims + 1
#         self._n_channels = n_channels
#         self._ssim_module = SSIM(data_range=data_range,
#                                  size_average=size_average,
#                                  win_size=window_size,
#                                  win_sigma=window_sigma,
#                                  channel=n_channels,
#                                  spatial_dims=spatial_dims,
#                                  K=k,
#                                  nonnegative_ssim=nonnegative_ssim)
        
#     def forward(self, input:Tensor, target:Tensor):
#         img_dims = input.shape
#         # Find batch size
#         batch_size = 1
#         if len(img_dims) > self._input_dims:
#             batch_size = img_dims[0]
#         s = self._ssim_module(input.view(batch_size, self._n_channels, img_dims[-2], img_dims[-1]),
#                               target.view(batch_size, self._n_channels, img_dims[-2], img_dims[-1]))
#         if self._is_loss:
#             return 1-s
#         return s
    
#     def __repr__(self):
#         _s = f"SSIMLoss(win_size={self._ssim_module.win_size})"
#         return _s
    
# if __name__ == "__main__":
#     import torch
#     ssiml = SSIMLoss(data_range=1.0, n_channels=1)
#     a = torch.tensor([[1,2.0]*20,[3.1,4.0]*20]*20)
#     b = torch.tensor([[1.2,1.9]*20,[3.1,4.1]*20]*20)
#     loss = ssiml(a,b)
#     print(loss.item())