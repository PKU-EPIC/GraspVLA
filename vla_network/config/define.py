import os
from os.path import join, dirname, abspath
from typing import Optional, List, Type, Union, Dict
from pydantic import BaseModel, Field
from PIL import Image
import torch
import importlib
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from vla_network.utils.path import get_path_exp
from vla_network.utils.logger import log

from enum import IntEnum


def optional_str(x: Union[str, None]) -> Union[str, None]:
    if x is None or x == "none" or x == "None":
        return None
    else:
        return x


class ImageTransform:
    def __call__(
        self, img: Image, **kwargs: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

class MixDatasetConfig(BaseModel):
    datasets: Optional[List[str]] = Field(default=None)
    datasets_type: Optional[List[str]] = Field(default=None)
    datasets_weight: Optional[List[float]] = Field(default=None)

    @staticmethod
    def from_str(string: str) -> "MixDatasetConfig":
        datasets = []
        datasets_type = []
        datasets_weight = []
        for item in string.split(";"):
            d, dt, dw = item.split(',')
            datasets.append(d)
            datasets_type.append(dt)
            datasets_weight.append(float(dw))
        return MixDatasetConfig(
            datasets=datasets,
            datasets_type=datasets_type,
            datasets_weight=datasets_weight,
        )


class STATE_REP(IntEnum):
    JOINT = 1
    EEF = 2


class ACTION_REP(IntEnum):
    REL_EEF = 1
    REL_JOINT = 2


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


class AssembledVLAModelConfig(BaseModel):
    backbone_2d: Backbone2DConfig
    llm: LLMConfig
    pred: str # flow_matching or token_pred
    action_expert: int
    action_expert_cfg: Optional[ActionExpertConfig] = None
    flow_matching_cfg: Optional[FlowMatchingConfig] = None

    backbone_2d_mode: str
    projector_mode: str
    llm_mode: str

    action_token_num: int


class VLAModelConfig(BaseModel):
    model_type: str

    # TODO: make this general to different model types
    model_specific: Optional[AssembledVLAModelConfig] = None

    ckpt: Optional[str] = None

    max_proprio_dim: int
    max_action_dim: int
    max_goal_dim: Optional[int] = Field(default=None)

    proprio_rep: Optional[STATE_REP] = None
    """Tells the dataset the desired representation of proprioception.
    We currently do not enforce, it's up to each dataset.
    """
    goal_rep: Optional[STATE_REP] = None
    """Like proprio_rep."""
    action_rep: Optional[ACTION_REP] = None
    """Tells the dataset the desired representation of actions.
    We currently do not enforce, it's up to each dataset.
    """

    dt: float
    """The time between each step."""
    proprio_steps: int
    action_steps: int
    image_steps: int
    image_keys: Optional[List[str]]
    """When this field is not None, the datasets should return images (and corresponding bboxs etc.) with the specified keys and orders.
    When this field is None, the keys and orders are decided by the dataset, but should generally ordered from most important to least important, from left to right.
    """
    image_size: Optional[int]
    """Only square images for now."""

    @property
    def img_num(self):
        return len(self.image_keys) * self.image_steps

    def to_dict(self):
        """Hugging Face Trainer/WandB expect a config object with to_dict()."""
        return self.model_dump()


class VLATrainConfig(BaseModel):
    exp_name: str
    args: Optional[TrainingArguments] = Field(init=False, default=None)
    max_steps: int
    global_batch_size: int
    device_batch_size: int
    lr: float
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    log_step: int
    save_step: int
    save_total_limit: int
    eval_step: int
    eval_each: int
    num_workers: int
    shuffle_buffer_size: int
    deepspeed: Optional[str]
    fsdp: Optional[str]
    profiler: bool
    bf16: bool
    full_bf16: bool
    gradient_checkpointing: bool
    allow_resume: bool = Field(default=False)

    reuse_preprocessor: bool = Field(default=True, description="Whether to reuse existing preprocessor")

    train_datasets: Union[str, MixDatasetConfig] = Field(default=MixDatasetConfig())
    val_datasets: Optional[Union[str, MixDatasetConfig]] = Field(default=None)

    normalizer_ratio_limit: float
    normalizer_num_samples: int

    def setup(self):
        if isinstance(self.train_datasets, str):
            self.train_datasets = MixDatasetConfig.from_str(self.train_datasets)
        if isinstance(self.val_datasets, str):
            self.val_datasets = MixDatasetConfig.from_str(self.val_datasets)

        # set batch size
        device_count = int(os.getenv("WORLD_SIZE", 1))
        assert self.global_batch_size % (self.device_batch_size * device_count) == 0, f'global_bs {self.global_batch_size} % (device_bs {self.device_batch_size} * device_count {device_count}) != 0'
        grad_accum = self.global_batch_size // (self.device_batch_size * device_count)
        log.info(
            f"Batch size {self.global_batch_size} = "
            + f"gpu num {device_count} * "
            + f"device batch size {self.device_batch_size} * "
            + f"grad_accum {grad_accum}"
        )

        # deepspeed
        deepspeed = optional_str(self.deepspeed)
        if deepspeed is not None:
            deepspeed = join(dirname(abspath(__file__)), deepspeed)

        self.args = TrainingArguments(
            output_dir=get_path_exp(self.exp_name),
            run_name=self.exp_name,
            deepspeed=deepspeed,
            max_steps=self.max_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            learning_rate=self.lr,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            per_device_train_batch_size=self.device_batch_size,
            per_device_eval_batch_size=self.device_batch_size,
            gradient_accumulation_steps=grad_accum,
            report_to="wandb" if self.exp_name != "debug" else "none",
            logging_strategy="steps",
            logging_steps=self.log_step,
            eval_strategy="steps",
            eval_steps=self.eval_step,
            eval_delay=self.eval_step,
            save_strategy="steps",
            save_steps=self.save_step,
            save_total_limit=self.save_total_limit,
            dataloader_drop_last=True,
            dataloader_num_workers=self.num_workers,
            dataloader_persistent_workers=False,
            # split_batches=True,
            # dispatch_batches=False,
            dataloader_prefetch_factor=32 if (self.num_workers >= 1) else None,
            fsdp=self.fsdp,
            fsdp_config=dict(
                limit_all_gathers=True,
                use_orig_params=True,
            ),
            seed=42+int(os.environ["RANK"])  # we have to manually set different seed for each rank, otherwise flow matching will sample identical noise
        )


class VLAConfig(BaseModel):
    model: VLAModelConfig
    train: VLATrainConfig
