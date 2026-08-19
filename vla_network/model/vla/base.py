from vla_network.datasample.base import BaseSample
from vla_network.config import VLAModelConfig

from transformers.modeling_outputs import ModelOutput

from abc import ABC, abstractmethod
from typing import List


class BaseVLA(ABC):
    """
    The base class of VLA models. In essense this interface is to accomodate for two common requirements in VLA:
    1. Diverse design choices of the model architecture;
    2. heterogeneous training samples, such as action, grounding, VQA.
    """
    def __init__(self, config: VLAModelConfig):
        """Construct the model based on the config, and load checkpoint if specified."""
        pass

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True):
        """Customize the requires_grad_() method.

        Typical cases: freeze vision encoder while unfreeze language backbone.
        """
        pass

    @abstractmethod
    def collate(self, batch: List[BaseSample]) -> dict:
        """The collated dict will be passed to the model's forward() method."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> ModelOutput:
        """Called during training.

        The return value only need to provide the loss, i.e., return ModelOutput(loss=xxx)."""
        pass

    @abstractmethod
    def predict(self, sample: BaseSample):
        """The model should predict based on the type of the sample."""
        pass

    def compile(self):
        """Compile the model for faster training/inference."""
        raise NotImplementedError()
