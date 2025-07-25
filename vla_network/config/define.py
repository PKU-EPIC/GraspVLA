from typing import Optional, List, Type, Union, Dict
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import torch
import importlib
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

class ImageTransform:
    def __call__(
        self, img: Image, **kwargs: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

# TODO: which keys should be in the basic config, shared in VA and VLA?
# Don't add keys without asking other people
class BasicDataConfig(BaseModel):
    exp_name: Optional[str] = Field(default=None)
    robot: str
    proprio_len: int
    action_len: int
    action_dim: int = Field(default=None)
    goal_dim: Optional[int] = Field(default=None)
    action_rel_len: int
    dt_steps: int
    
    def setup(self):
        pass


class VLADataConfig(BasicDataConfig):
    # TODO: sort them in a better way
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Optional[PreTrainedTokenizerBase] = Field(init=False, default=None)
    image_transform: Optional[ImageTransform] = Field(init=False, default=None)
    action_token_num: int
    img_steps: int
    img_key: Optional[List[str]]
    image_size: Optional[int]
    anything_prob: float
    robot_rep: str
    goal_rep: Optional[str]
    tokenizer_type: str
    tokenizer_ratio_limit: float
    count_num: int
    trans_noise: float
    rot_noise: float
    brightness_img: str
    brightness_threshold: float
    crop_mode: Dict[str, str]
    proprio_dim: Optional[int] = Field(default=None)
    use_bbox: int
    pred: Optional[str] = Field(init=False, default=None)

    def setup(self):
        super().setup()

        if self.action_dim is None:
            if self.robot_rep in ['xyz_rpy', 'xyz_rpy_rot']:
                self.action_dim = 7 # xyz rpy gripper
        if self.goal_dim is None:
            if self.goal_rep == 'xyz_rpy':
                self.goal_dim = 6 # xyz rpy
            elif self.goal_rep == 'xyz_rot':
                self.goal_dim = 12 # xyz rotmat
        if self.proprio_dim is None:
            if self.robot_rep == 'xyz_rpy':
                self.proprio_dim = 7 # xyz rpy gripper
            elif self.robot_rep == 'xyz_rpy_rot':
                self.proprio_dim = 13 # xyz rotmat gripper

    
    @property
    def img_num(self) -> int:
        return len(self.img_key) * self.img_steps


LLM_CONFIG = {
    "meta-llama/Llama-2-7b-hf": {
        "family": "llama2",
        "model_cls": ("transformers", "LlamaForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
    "internlm/internlm2-1_8b": {
        "family": "internlm",
        "model_cls": (
            "vla_network.model.backbone_llm.internlm.modeling_internlm2",
            "InternLM2ForCausalLM",
        ),
        "token_cls": (
            "vla_network.model.backbone_llm.internlm.tokenization_internlm2_fast",
            "InternLM2TokenizerFast",
        ),
    },
}


# TODO: which keys should be in the basic config, shared in VA and VLA?
# Don't add keys without asking other people
class BasicModelConfig(BaseModel):
    pass

class LLMConfig(BaseModel):
    name: str
    max_len: int = Field(default=2048)
    special_tokens: List[str] = Field(default_factory=lambda: [])
    pad_multiple_of: int = Field(default=64)
    attn_implementation: str

    @property
    def family(self) -> str:
        return LLM_CONFIG[self.name]["family"]

    @staticmethod
    def get_cls(package: str, name: str):
        module = importlib.import_module(package)
        return getattr(module, name)

    @property
    def model_cls(self) -> Type[PreTrainedModel]:
        cls_package, cls_name = LLM_CONFIG[self.name]["model_cls"]
        return self.get_cls(cls_package, cls_name)

    @property
    def token_cls(self) -> Type[PreTrainedTokenizerFast]:
        cls_package, cls_name = LLM_CONFIG[self.name]["token_cls"]
        return self.get_cls(cls_package, cls_name)


class Backbone2DConfig(BaseModel):
    name: str
    image_size: int

class ActionExpertConfig(BaseModel):
    hidden_size_scale: Optional[int] = Field(default=None)
    intermediate_size_scale: Optional[int] = Field(default=None)
    hidden_size: Optional[int] = Field(init=False, default=None)
    intermediate_size: Optional[int] = Field(init=False, default=None)
    hidden_act: Optional[str] = Field(init=False, default=None)

class FlowMatchingConfig(BaseModel):
    beta_alpha: float
    beta_beta: float
    time_min: float
    time_max: float

class VLAModelConfig(BasicModelConfig):
    backbone_2d: Backbone2DConfig
    llm: LLMConfig
    ckpt: str
    pred: str # flow_matching or token_pred
    action_len: int = Field(init=False, default=None)
    action_dim: int = Field(init=False, default=None)
    proprio_dim: int = Field(init=False, default=None)
    action_expert: int
    action_expert_cfg: Optional[ActionExpertConfig] = None
    flow_matching_cfg: Optional[FlowMatchingConfig] = None

    def to_dict(self):
        return self.model_dump()


class BasicConfig(BaseModel):
    data: BasicDataConfig
    model: BasicModelConfig
    dummy: str = None  # For unneeded arguments


class VLAConfig(BasicConfig):
    data: VLADataConfig
    model: VLAModelConfig
    dummy: str = None  # For unneeded arguments