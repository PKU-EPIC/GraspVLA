from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict, BeforeValidator, PlainSerializer
from typing_extensions import Annotated
import numpy as np
import torch

def nd_array_custom_before_validator(x):
    # custom before validation logic for np.ndarray
    return np.array(x)


def nd_array_custom_serializer(x):
    # custom serialization logic for np.ndarray
    return x.tolist()

NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=list, when_used="json"),
] # A wrapper of np.ndarray in order to make it usable in pydantic

class RawVLAData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # instruction
    instruction: Optional[str] = None
    can_be_anything: bool = False

    # Observation
    images: Optional[Dict[str, NdArray]] = None
    bboxs: Optional[Dict[str, NdArray]] = None
    pcs: Optional[Dict[str, NdArray]] = None
    proprio: NdArray = None
    proprio_flag: NdArray = None
    for_rel_proprio: NdArray = None
    for_rel_proprio_flag: NdArray = None

    # Action
    action: Optional[NdArray] = None
    action_flag: Optional[NdArray] = None
    goal: Optional[NdArray] = None  
    goal_trans: Optional[NdArray] = None
    goal_rot: Optional[NdArray] = None  


@dataclass
class BatchVLAData:
    debug: List[Any]  # TODO: Conflict with huggingface now
    # With the following things:
    # dataset_name: List[[str]]
    # data_id: List[[str]]
    # orig_instruction: List[[str]]
    # instruction: List[[str]]

    # tokens
    input_ids: torch.Tensor  # (B, N_token)
    robot_input_ids: torch.Tensor # (B, N_robot_token)
    labels: Optional[torch.Tensor]  # (B, N_token)
    robot_labels: Optional[torch.Tensor] # (B, N_robot_token)
    attention_mask: torch.Tensor  # (B, N_token)
    robot_attention_mask: torch.Tensor # (B, N_robot_token)

    # robot
    action: torch.Tensor # (B, T_action, D_action)
    proprio: torch.Tensor # (B, T_proprio, D_proprio)
    goal: Optional[torch.Tensor] # (B, D_goal)

    # Images
    images: torch.Tensor  # (B, T_image, N_backbone, C, H, W)

    # type
    is_action: torch.Tensor  # (B,)

    # inference
    inference_kwargs: Optional[list] = None