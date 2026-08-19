from .base import BaseSample

from pydantic import ConfigDict
from typing import Optional, Dict, List, OrderedDict
from PIL import Image
from vla_network.type import NdArray


class VLASample(BaseSample):
    """
    A data sample for vision-languange-action training.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embodiment: str
    """A string that distinguishes the embodiment, can be used in normalization key, prompt, etc."""

    instruction: Optional[str] = None
    """Full instruction text"""
    frame: Optional[int] = None
    """The frame id within the trajectory."""

    # Observation
    images: Optional[OrderedDict[str, List[Image.Image]]] = None
    """each of shape (T_image, H, W, C)"""
    bboxs: Optional[OrderedDict[str, List[NdArray]]] = None
    """camera_name: bbox (T_image, 4), XYXY"""

    proprio: NdArray = None
    """proprio joint angle (T_proprio, D_proprio)"""
    goal: Optional[NdArray] = None  
    """(D_goal,)"""
    action: Optional[NdArray] = None
    """(T_action, D_action)"""
