"""
Generic DL models which can be adapted or modified to create neural networks
"""
from oil.models.mpl import MLP
from oil.models.fourier_features import FourierFeatureMLP, FourierFeatures
from oil.models.siren import SirenLayer, SirenNet