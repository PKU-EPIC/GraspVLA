from abc import ABC, abstractmethod
import torch
from torch import nn

from vla_network.config import Backbone2DConfig, ImageTransform


class Backbone2D(nn.Module, ABC):
    config: Backbone2DConfig
    image_transform: ImageTransform

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init(config: Backbone2DConfig) -> "Backbone2D":
        if config.name == "dinosiglip":
            from .dinosiglip_vit import DinoSigLIPViTBackbone

            return DinoSigLIPViTBackbone(config)
        else:
            raise NotImplementedError
