# FNETs

This repo contains the code, data and weights used in the paper: [_High-Fidelity Prediction of Perturbed Optical Fields using Fourier Feature Networks_](https://arxiv.org/abs/2508.19751).

## Citation
All python code and model weights are free to used under a standard MIT licence.

Please cite our associated paper (available on ArXiV):

```
@article{jandrell2025fnet,
  title        = {High-Fidelity Prediction of Perturbed Optical Fields using Fourier Feature Networks},
  author       = {Jandrell, Joshua R. and Cox, Mitchell A.},
  year         = {2025},
  month        = sep,
  journal      = {arXiv preprint},
  volume       = {arXiv:2508.19751},
  doi          = {10.48550/arXiv.2508.19751},
  url          = {https://arxiv.org/abs/2508.19751}
}
```
> [!IMPORTANT]
> Licencing and Citation does not cover the training data (`.h5`) files.
> These files contain reformatted data originally  collected by M. W. Matthès, Y. Bromberg, J. de Rosny and S. M. Popoff for their paper [_Learning and Avoiding Disorder in Multimode Fibers_](https://arxiv.org/abs/2010.14813).
> 
> Original data can be retrieved from [this repo](https://github.com/wavefrontshaping/article_MMF_disorder).
>

## ⚡Quick Demo
For a brief demonstration of how to load model weights and plots of how well model predictions match data see [`plot_fits.ipynb`](plot_fits.ipynb)

## Pre-Trained Model Weights 
Weights for pretrained pytorch models can be found in the [`Data/Weights`](Data/Weights) directory.

You can use these files to load full pytorch models without including or importing any additional code:
```python
import torch
model = torch.load(<path-to-weights> weights_only=False)
```
> [!TIP]
> All models predict real and imaginary complex components as two concatenated tensors of floating point values.
> The custom [`oil/complex`](oil/complex) module can be used to convert these into a single tensor of `torch.complex` values.
> ```python
> from oil.complex import ComplexConverter
> re_im_cc = ComplexConverter(expand_method='concat',representation='real-imag')
> raw_pred = model(x)
> complex_pred = re_im_cc.float_to_complex(raw_pred)
> ```
> 

## Additional Source Code 
Source code for a Fourier feature pytorch module, neural networks models and other useful custom pytorch code can be found in the custom (`oil`)[oil] module. Feel free to use this code as you wish! We hope you find it helpful (please cite our paper if you do).
The oil module contains some scripts we wrote to make pytorch run a bit smoother and be a bit simpler to use. The version included in this repository is incomplete and is provided (without warranty) under the this repo’s general MIT license.
