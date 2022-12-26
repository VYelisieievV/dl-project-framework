import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Dict, Callable, Union, Any


__doc__ = "Here we define different Lightning models which can be used."


class BaseLightningModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)


__all__ = [BaseLightningModel]
