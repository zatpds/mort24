from typing import Union

from mort24.models.dl_models.rnn import GRUNet, LSTMNet, RNNet
from mort24.models.dl_models.tcn import TemporalConvNet
from mort24.models.dl_models.transformer import BaseTransformer, LocalTransformer, Transformer

DLModel = Union[
    GRUNet,
    RNNet,
    LSTMNet,
    TemporalConvNet,
    BaseTransformer,
    Transformer,
    LocalTransformer,
]

__all__ = [
    "GRUNet",
    "RNNet",
    "LSTMNet",
    "TemporalConvNet",
    "BaseTransformer",
    "Transformer",
    "LocalTransformer",
    "DLModel",
]
