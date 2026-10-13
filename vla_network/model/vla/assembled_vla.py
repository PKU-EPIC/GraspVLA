from .base import BaseVLA

from torch.nn.utils.rnn import pad_sequence
from functools import singledispatchmethod

from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any, Tuple
import numpy as np
import copy
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import ModelOutput

from vla_network.utils.logger import log
from vla_network.type import NdArray

from vla_network.datasample.base import BaseSample
from vla_network.datasample import VLASample

from vla_network.model.backbone_2d import Backbone2D
from vla_network.model.backbone_llm import LLMBackbone
from vla_network.config import VLAModelConfig, ActionExpertConfig
from vla_network.utils.constant import IGNORE_INDEX

from vla_network.model.common.projector import FusedMLPProjector
from vla_network.model.common.flow_matching import VLAFlowMatchingModule

from vla_network.utils.token_pattern import TokenInfo, TokenPattern, TokenResult
from vla_network.utils.tokenizer import UniformTokenizer


COT_PROMPT = lambda prompt: f"In: What action should the robot take to {prompt}?\nOut: "


def make_block_attn_mask(input_mask, block_mask):
    cumsum = torch.cumsum(block_mask, dim=0)
    causal_num = (cumsum == 0).sum()
    causal_mask = torch.tril(torch.ones((input_mask.shape[1], input_mask.shape[1]), dtype=torch.bool, device=input_mask.device))
    if causal_num != len(block_mask):
        block_attn_mask = cumsum[None, causal_num:] <= cumsum[causal_num:, None]
        causal_mask[causal_num:, causal_num:] = block_attn_mask
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(causal_mask, valid_mask)[:, None]


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


def get_graspvla_pattern(config: VLAModelConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key='text_ids', length=None, est=False, as_input=True),
            TokenInfo(key='bbox', length=config.img_num * 4, est=True, as_input=False),
        ],
        robot_infos=[
            TokenInfo(key='hist_proprio', length=(config.proprio_steps-1) * config.max_proprio_dim, est=False, as_input=True),
            TokenInfo(key='cur_proprio', length=config.max_proprio_dim, est=False, as_input=True),
            TokenInfo(key='goal', length=config.max_goal_dim, est=True, as_input=False),
            TokenInfo(key='action', length=config.action_steps*config.max_action_dim, est=True, as_input=False),
            TokenInfo(key='eos', length=1, est=True, as_input=False),
        ]
    )

def get_pi0_pattern(config: VLAModelConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key='text_ids', length=None, est=False, as_input=True),
        ],
        robot_infos=[
        ],
    )

def get_pi0_cot_action_pattern(config: VLAModelConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key='text_ids', length=None, est=False, as_input=True),
            TokenInfo(key='bbox', length=config.img_num * 4, est=True, as_input=False),
            TokenInfo(key='hist_proprio', length=(config.proprio_steps-1) * config.max_proprio_dim, est=False, as_input=True),
            TokenInfo(key='cur_proprio', length=config.max_proprio_dim, est=False, as_input=True),
            TokenInfo(key='goal', length=config.max_goal_dim, est=True, as_input=False),
            TokenInfo(key='eos', length=1, est=True, as_input=False),
        ],
        robot_infos=[
        ],
    )

def get_pi0_bbox_cot_pattern(config: VLAModelConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key='text_ids', length=None, est=False, as_input=True),
            TokenInfo(key='bbox', length=config.img_num * 4, est=True, as_input=False),
            TokenInfo(key='eos', length=1, est=True, as_input=False),
        ],
        robot_infos=[
        ],
    )

# fmt: on

def get_token_pattern(config: VLAModelConfig, name: str) -> TokenPattern:
    return dict(
        graspvla=get_graspvla_pattern,
        pi0=get_pi0_pattern,
        pi0_cot_action=get_pi0_cot_action_pattern,
        pi0_bbox_cot=get_pi0_bbox_cot_pattern,
    )[name](config)


class AssembledVLA(nn.Module, BaseVLA):
    config: VLAModelConfig
    backbone_2d: Backbone2D
    llm: LLMBackbone
    projector: nn.Module
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, config: VLAModelConfig):
        super().__init__()
        self.config = config

        self.backbone_2d = Backbone2D.init(self.config.model_specific.backbone_2d)
        self.backbone_2d_dim = self.backbone_2d.feature_dim

        self.llm = LLMBackbone(self.config.model_specific.llm)
        self.llm_dim = self.llm.input_dim

        self.projector = FusedMLPProjector(self.backbone_2d_dim, self.llm_dim)

        self.uniform_tokenizer = UniformTokenizer(self.config.model_specific.action_token_num, self.llm.tokenizer.vocab_size)

        if self.config.model_specific.pred in ["flow_matching", "cot_flow_matching", "cot_bbox_flow_matching"]:
            self.action_expert = self._create_action_expert_from_llm(self.llm.llm, self.config.model_specific.action_expert_cfg)
            self.flow_module = VLAFlowMatchingModule(
                config=self.config.model_specific.flow_matching_cfg,
                action_dim=self.config.max_action_dim,
                llm_dim=self.action_expert.config.hidden_size,
                action_len=self.config.action_steps,
                proprio_dim=self.config.max_proprio_dim,
            )

        if config.model_specific.pred == 'token_pred':
            self.action_pattern = get_token_pattern(config, 'graspvla')
        elif config.model_specific.pred == 'flow_matching':
            self.action_pattern = get_token_pattern(config, 'pi0')
        elif config.model_specific.pred == 'cot_flow_matching':
            self.action_pattern = get_token_pattern(config, 'pi0_cot_action')
        elif config.model_specific.pred == 'cot_bbox_flow_matching':
            self.action_pattern = get_token_pattern(config, 'pi0_bbox_cot')
        else:
            raise ValueError(
                f"Unsupported prediction mode: {config.model_specific.pred}. "
                "Expected one of token_pred, flow_matching, cot_flow_matching, "
                "or cot_bbox_flow_matching."
            )

        log.info("Initialized VLA model")

        if config.ckpt is not None:
            self._load_pretrained(config.ckpt)

    def _update_state_dict_for_flat(self, state_dict: dict) -> dict:
        # support flat prefix keys by extracting them into sub-dicts
        def _extract_prefix(sd: dict, prefix: str) -> dict:
            pre = prefix + "."
            ret = {}
            # collect then remove to avoid changing dict while iterating
            to_remove = [k for k in sd.keys() if k.startswith(pre)]
            for k in to_remove:
                ret[k[len(pre):]] = sd.pop(k)
            return ret

        for p in ["llm", "backbone_2d", "projector", "flow_module"]:
            if p not in state_dict and any(k.startswith(p + ".") for k in state_dict.keys()):
                sub = _extract_prefix(state_dict, p)
                if len(sub) > 0:
                    state_dict[p] = sub
        return state_dict

    def _load_pretrained(self, path: str):
        # Decide loader by file extension; no exception fallbacks per request
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors  # type: ignore
            state_dict = load_safetensors(path)
            state_dict = self._update_state_dict_for_flat(state_dict)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)

        if "backbone_2d" in state_dict:
            self.backbone_2d.load_state_dict(state_dict["backbone_2d"])
        self.projector.load_state_dict(state_dict["projector"])
        llm_load_warn = self.llm.load_state_dict(state_dict["llm"], strict=False)
        log.info(f"Loaded model from {path}")
        if len(llm_load_warn.missing_keys) > 0 or len(llm_load_warn.unexpected_keys) > 0:
            log.warn(f'LLM load warning: {llm_load_warn}')
        
        # Load flow module with support for enlarged dimensions
        flow_state_dict = state_dict["flow_module"]
        model_flow_state_dict = self.flow_module.state_dict()
        
        for k, v in flow_state_dict.items():
            if k in model_flow_state_dict:
                if v.shape != model_flow_state_dict[k].shape:
                    # Check if the new shape is larger than or equal to the old shape in all dimensions
                    if all(s1 >= s2 for s1, s2 in zip(model_flow_state_dict[k].shape, v.shape)):
                        log.warn(f"Shape mismatch for flow_module.{k}: checkpoint {v.shape}, model {model_flow_state_dict[k].shape}. "
                                 f"Loading partial weights.")
                        # Create a slice object to copy the weights
                        slices = tuple(slice(0, s) for s in v.shape)
                        model_flow_state_dict[k][slices] = v
                    else:
                        log.warn(f"Shape mismatch for flow_module.{k}: checkpoint {v.shape}, model {model_flow_state_dict[k].shape}. "
                                 f"Cannot load weights due to shrunk dimensions.")
                else:
                    model_flow_state_dict[k] = v
            else:
                log.warn(f"Key {k} in checkpoint flow_module but not in model.")
        
        self.flow_module.load_state_dict(model_flow_state_dict)

    @staticmethod
    def _create_action_expert_from_llm(llm: PreTrainedModel, action_expert_config: ActionExpertConfig):
        config = copy.deepcopy(llm.config)
        config.hidden_size = config.hidden_size // action_expert_config.hidden_size_scale
        config.intermediate_size = config.intermediate_size // action_expert_config.intermediate_size_scale
        config.hidden_act = config.hidden_act
        config.head_dim = llm.model.layers[0].attention.head_dim
        model_cls = type(llm)
        return model_cls._from_config(config)

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        if requires_grad:
            for m in ["backbone_2d", "llm", "projector"]:
                mode = getattr(self.config.model_specific, f"{m}_mode")
                if mode == "train":
                    getattr(self, m).requires_grad_(True)
                elif mode == "freeze":
                    getattr(self, m).requires_grad_(False)
                else:
                    raise NotImplementedError()

        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_train_params / 10**6:.3f} Trainable"
        )

    @staticmethod
    def insert_img_info(orig: torch.Tensor, img_info: torch.Tensor) -> torch.Tensor:
        return torch.cat([orig[:, :1], img_info, orig[:, 1:]], dim=1) # fmt: skip

    @staticmethod
    def insert_img_info_single(orig: torch.Tensor, img_info: torch.Tensor) -> torch.Tensor:
        return torch.cat([orig[:1], img_info, orig[1:]], dim=0) # fmt: skip
    
    def get_proj_feat_2d(self, images: torch.FloatTensor) -> torch.FloatTensor:
        with torch.set_grad_enabled(False):
            feat_2d = self.backbone_2d(images)
        proj_feat_2d = self.projector(feat_2d)
        return proj_feat_2d

    def embed_prefix(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        images: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        proj_feat_2d: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
        
        b = len(input_ids)
        if proj_feat_2d is None:
            proj_feat_2d = self.get_proj_feat_2d(images)
        n_img_token = proj_feat_2d.shape[1]

        input_embed = self.llm.input_embedding(input_ids)
        mm_input_embed = self.insert_img_info(input_embed, proj_feat_2d).to(
            input_embed.dtype
        )

        img_attn_mask = torch.ones(
            (b, n_img_token), dtype=torch.bool, device=attention_mask.device
        )
        mm_attn_mask = self.insert_img_info(attention_mask, img_attn_mask)

        n_mm_token = mm_attn_mask.shape[1]
        mm_block_mask = torch.zeros(
            (n_mm_token, ), dtype=torch.bool,
            device=attention_mask.device
        )
        
        if labels is None:
            mm_labels = None
        else:
            img_labels = torch.full(
                (b, n_img_token), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            mm_labels = self.insert_img_info(labels, img_labels)
        
        return mm_input_embed, mm_attn_mask, mm_block_mask, mm_labels
    
    def embed_suffix_token_pred(
        self, 
        robot_input_ids: torch.LongTensor = None,
        robot_attention_mask: torch.Tensor = None,
        robot_labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
        
        if robot_input_ids.shape[-1] > 0:
            robot_input_embed = self.llm.input_embedding(robot_input_ids)
        else:
            robot_input_embed = torch.zeros(0, device=robot_input_ids.device, dtype=self.llm.input_embedding.weight.dtype)
        
        n_robot_token = robot_attention_mask.shape[1]
        robot_block_mask = torch.zeros(
            (n_robot_token, ), dtype=torch.bool,
            device=robot_attention_mask.device
        )
        
        return robot_input_embed, robot_attention_mask, robot_block_mask, robot_labels

    def _process_images_and_bboxs(self, images: Dict[str, NdArray], bboxs: Optional[Dict[str, NdArray]]):
        """Flatten and normalize images and bboxes."""
        ret_pixel_values = []
        ret_bboxes = []
        for i in range(self.config.image_steps):
            for key in self.config.image_keys:
                ret_pixel_values.append(self.backbone_2d.image_transform(images[key][i]))
                width, height = images[key][i].size
                ret_bboxes.append(bboxs[key][i] / np.array([width, height]*2) * 2 - 1 if bboxs is not None else None)
        ret_pixel_values = torch.stack(ret_pixel_values)
        ret_bboxes = np.stack(ret_bboxes) if bboxs is not None else None
        return ret_pixel_values, ret_bboxes

    @singledispatchmethod
    def _unify_sample(sample, inference: bool) -> BatchVLAData:
        pass

    @_unify_sample.register(VLASample)
    def _(self, sample: VLASample, inference: bool) -> BatchVLAData:
        text_ids = self.llm.tokenizer(COT_PROMPT(sample.instruction), add_special_tokens=True).input_ids
        images, bboxs = self._process_images_and_bboxs(sample.images, sample.bboxs)
        trans_dic = {
            'proprio': sample.proprio,
            'action': sample.action if hasattr(sample, 'action') else None,
            'goal': sample.goal if hasattr(sample, 'goal') else None
        }

        debug_dict = None
        if not inference:
            assert len(trans_dic["action"]) == self.config.action_steps

            input_ids, labels = self.action_pattern.get_input_id_label(
                text_ids=text_ids,
                bbox=self.uniform_tokenizer.uniform_tokenize(bboxs) if bboxs is not None else None,
                goal=self.uniform_tokenizer.uniform_tokenize(trans_dic['goal']) if 'goal' in trans_dic else None,
                hist_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][:-1]) if 'proprio' in trans_dic else None,
                cur_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][-1]) if 'proprio' in trans_dic else None,
                eos=[self.llm.tokenizer.eos_token_id],
            )

            robot_input_ids, robot_labels = self.action_pattern.get_robot_input_id_label(
                hist_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][:-1]),
                cur_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][-1]),
                goal=self.uniform_tokenizer.uniform_tokenize(trans_dic['goal']) if trans_dic['goal'] is not None else None,
                action=self.uniform_tokenizer.uniform_tokenize(trans_dic['action']),
                eos=[self.llm.tokenizer.eos_token_id],
            )
            inference_kwargs = None
        else:
            inference_kwargs = [dict(
                text_ids=text_ids,
                hist_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][:-1]),
                cur_proprio=self.uniform_tokenizer.uniform_tokenize(trans_dic['proprio'][-1]),
            )]
            token_result = self.action_pattern.update_tokens(
                output=[], 
                **inference_kwargs[0]
            )
            input_ids = token_result.input_ids
            robot_input_ids = token_result.robot_input_ids
            if 'action' in trans_dic:
                debug_dict = dict(
                    action=trans_dic['action'],
                    goal=trans_dic['goal'] if trans_dic.get('goal') is not None else None
                )

        return BatchVLAData(
            debug=[debug_dict],
            input_ids=torch.tensor(input_ids)[None],
            labels=torch.tensor(labels)[None] if not inference else None,
            attention_mask=torch.ones(len(input_ids))[None].bool(),
            robot_input_ids=torch.tensor(robot_input_ids)[None],
            robot_attention_mask=torch.ones(len(robot_input_ids))[None].bool(),
            robot_labels=torch.tensor(robot_labels)[None] if not inference else None,
            images=images[None],
            action=torch.from_numpy(trans_dic['action']).float()[None] if trans_dic['action'] is not None else None,
            proprio=torch.from_numpy(trans_dic['proprio']).float()[None],
            goal=torch.from_numpy(trans_dic['goal']).float()[None] if trans_dic['goal'] is not None else None,
            is_action=torch.ones(1).bool(),
            inference_kwargs=inference_kwargs,
        )

    def collate(self, batch: List[BaseSample]):
        datas: List[BatchVLAData] = [self._unify_sample(b, inference=False) for b in batch]

        kwargs = dict()

        pad_idx = self.llm.tokenizer.pad_token_id
        max_len = self.llm.tokenizer.model_max_length
        input_ids = pad_sequence(
            [data.input_ids[0] for data in datas],
            batch_first=True,
            padding_value=pad_idx,
        )
        robot_input_ids = pad_sequence(
            [data.robot_input_ids[0] for data in datas],
            batch_first=True,
            padding_value=pad_idx,
        )
        log.warn_when(
            input_ids.shape[1] > max_len,
            f"input_ids length = {input_ids.shape[1]} > {max_len}",
        )
        kwargs["input_ids"] = input_ids[:, :max_len]
        kwargs["robot_input_ids"] = robot_input_ids
        kwargs["attention_mask"] = kwargs["input_ids"] != pad_idx
        kwargs["robot_attention_mask"] = robot_input_ids != pad_idx
        if datas[0].labels is not None:
            labels = pad_sequence(
                [data.labels[0] for data in datas],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            kwargs["labels"] = labels[:, :max_len]
            robot_labels = pad_sequence(
                [data.robot_labels[0] for data in datas],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            kwargs["robot_labels"] = robot_labels

        for k in ['images', 'action', 'proprio', 'goal', 'is_action']:
            if getattr(datas[0], k, None) is not None:
                kwargs[k] = torch.cat([getattr(data, k) for data in datas], dim=0)
            else:
                kwargs[k] = None
        return kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        robot_input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        robot_attention_mask: torch.Tensor = None,
        images: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        robot_labels: Optional[torch.LongTensor] = None,
        action: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_action: Optional[torch.BoolTensor] = None,
        debug: List[Any] = None,
    ) -> Union[CausalLMOutputWithPast, ModelOutput]:
    
        prefix_embeds, prefix_mask, prefix_block_mask, prefix_labels = self.embed_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels
        )
        
        if self.config.model_specific.pred == "flow_matching":
            action, proprio, goal = action.to(prefix_embeds.dtype), proprio.to(prefix_embeds.dtype), goal.to(prefix_embeds.dtype)
            x_t, u_t, time = self.flow_module.sample_noise_and_time(action)
            suffix_embeds, suffix_mask, suffix_block_mask = self.flow_module.embed_suffix_flow_matching(
                proprio=proprio,
                noisy_actions=x_t,
                timestep=time
            )
            full_labels = None
        elif self.config.model_specific.pred in ["cot_flow_matching", "cot_bbox_flow_matching"]:
            action, proprio = action.to(prefix_embeds.dtype), proprio.to(prefix_embeds.dtype)
            x_t, u_t, time = self.flow_module.sample_noise_and_time(action)
            suffix_embeds, suffix_mask, suffix_block_mask = self.flow_module.embed_suffix_flow_matching(
                proprio=proprio,
                noisy_actions=x_t,
                timestep=time
            )
            full_labels = prefix_labels
        elif self.config.model_specific.pred == "token_pred":
            suffix_embeds, suffix_mask, suffix_block_mask, suffix_labels = self.embed_suffix_token_pred(
                robot_input_ids=robot_input_ids,
                robot_attention_mask=robot_attention_mask,
                robot_labels=robot_labels
            )
            full_labels = torch.cat((prefix_labels, suffix_labels), dim=1)
        else:
            raise ValueError(f"Unsupported prediction mode: {self.config.model_specific.pred}")
        
        full_input_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        if self.config.model_specific.llm.attn_implementation == "sdpa":
            block_mask = torch.cat((prefix_block_mask, suffix_block_mask), axis=0)
            full_attn_mask = make_block_attn_mask(full_input_mask, block_mask).to(prefix_embeds.dtype)
        else:
            full_attn_mask = full_input_mask
        full_position_ids = torch.cumsum(full_input_mask, dim=1) - 1       

        prefix_len = prefix_embeds.shape[1]
        if self.config.model_specific.action_expert:
            llm_output = self.llm(
                inputs_embeds=prefix_embeds,
                attention_mask=full_attn_mask[:, :, :prefix_len, :prefix_len] if self.config.model_specific.llm.attn_implementation == "sdpa" else full_attn_mask[:, :prefix_len],
                position_ids=full_position_ids[:, :prefix_len],
                labels=None if full_labels is None else full_labels[:, :prefix_len],
                use_cache=True,
                output_hidden_states=True,
            )
            action_expert_output = self.action_expert(
                inputs_embeds=suffix_embeds,     
                attention_mask=full_attn_mask[:, :, prefix_len:] if self.config.model_specific.llm.attn_implementation == "sdpa" else full_attn_mask[:, prefix_len:],
                position_ids=full_position_ids[:, prefix_len:],
                labels=(
                    None
                    if full_labels is None or prefix_len == full_labels.shape[1]
                    else full_labels[:, prefix_len:]
                ),
                use_cache=True,
                output_hidden_states=True,
                past_key_values=llm_output.past_key_values,
            )
        else:
            full_input_embed = torch.cat((prefix_embeds, suffix_embeds), dim=1)
            llm_output = self.llm(
                inputs_embeds=full_input_embed,
                attention_mask=full_attn_mask,
                position_ids=full_position_ids,
                labels=full_labels,
                use_cache=False,
                output_hidden_states=True,
            )
        
        if self.config.model_specific.pred == "flow_matching":
            if self.config.model_specific.action_expert:
                action_hidden_states = action_expert_output.hidden_states[-1][:, -action.shape[-2]:]
            else:
                action_hidden_states = llm_output.hidden_states[-1][:, -action.shape[-2]:]
            loss = torch.sum(self.flow_module.compute_loss(action_hidden_states, u_t) * is_action) / torch.clip(is_action.sum(), 1, None)
            output = ModelOutput(loss=loss)
        elif self.config.model_specific.pred in ["cot_flow_matching", "cot_bbox_flow_matching"]:
            if self.config.model_specific.action_expert:
                action_hidden_states = action_expert_output.hidden_states[-1][:, -action.shape[-2]:]
            else:
                action_hidden_states = llm_output.hidden_states[-1][:, -action.shape[-2]:]
            flow_matching_loss = torch.sum(self.flow_module.compute_loss(action_hidden_states, u_t) * is_action) / torch.clip(is_action.sum(), 1, None)
            output = ModelOutput(loss=flow_matching_loss + llm_output.loss)
        else:
            output = llm_output
        return output

    @singledispatchmethod
    def predict(self, sample: BaseSample) -> BaseSample:
        pass
    @predict.register(VLASample)
    def _(self, sample: VLASample) -> VLASample:
        model_input = self._unify_sample(sample, inference=True)
        device = next(self.parameters()).device
        token_result, action_result = self.generate(
            input_ids=model_input.input_ids.to(device),
            robot_input_ids=model_input.robot_input_ids.to(device),
            attention_mask=model_input.attention_mask.to(device),
            robot_attention_mask=model_input.robot_attention_mask.to(device),
            images=model_input.images.to(device),
            proprio=model_input.proprio.to(device),
            inference_kwargs=model_input.inference_kwargs,
            token_pattern=self.action_pattern,
            max_token_num=100,
        )
        ret = {}
        if self.config.model_specific.pred == "flow_matching":
            ret['action'] =  action_result.float().cpu().numpy()[0]
        elif self.config.model_specific.pred in ["cot_flow_matching", "cot_bbox_flow_matching"]:
            ret['action'] = action_result.float().cpu().numpy()[0]
            if hasattr(token_result, 'goal'):
                ret['goal'] = self.uniform_tokenizer.uniform_detokenize(np.array(token_result.goal))
            if hasattr(token_result, 'bbox'):
                assert self.config.image_steps == 1
                ret['bboxs'] = {}
                for idx, key in enumerate(self.config.image_keys):
                    ret['bboxs'][key] = [(self.uniform_tokenizer.uniform_detokenize(np.array(token_result.bbox).reshape(-1, 4)[idx]) + 1)/2*np.array([sample.images[key][-1].width, sample.images[key][-1].height]*2)]
        elif self.config.model_specific.pred == 'token_pred':
            ret['action'] = np.array(token_result.action).reshape(-1, self.config.max_action_dim)
            if hasattr(token_result, 'goal'):
                ret['goal'] = self.uniform_tokenizer.uniform_detokenize(np.array(token_result.goal))
            if hasattr(token_result, 'bbox'):
                assert self.config.image_steps == 1
                ret['bboxs'] = {}
                for idx, key in enumerate(self.config.image_keys):
                    ret['bboxs'][key] = [(self.uniform_tokenizer.uniform_detokenize(np.array(token_result.bbox).reshape(-1, 4)[idx]) + 1)/2*np.array([sample.images[key][-1].width, sample.images[key][-1].height]*2)]
        else:
            raise NotImplementedError()
        return sample.model_copy(update=ret)

    def compile(self):
        self.llm.llm = torch.compile(self.llm.llm, dynamic=True)
        self.backbone_2d = torch.compile(self.backbone_2d)
        if hasattr(self, 'action_expert'):
            self.action_expert = torch.compile(self.action_expert, dynamic=True)

    # TODO: remove unused inputs
    # TODO: what should be the output type of this function?
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        robot_input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        robot_attention_mask: torch.Tensor = None,
        images: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        proprio: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        debug: List[Any] = None,
        # TODO: maybe requires runtime config
        max_token_num: int = int(1e10),
        flow_matching_iter: int = 10,
        inference_kwargs: List[dict] = None,
        token_pattern: Optional[TokenPattern] = None,
    ) -> Tuple[TokenResult, Any]:
        # TODO: This is a temporary solution
        # Latter we will change to C++ implementation
        # So don't care about the performance
    
        proj_feat_2d = self.get_proj_feat_2d(images)
        prefix_embeds, prefix_mask, prefix_block_mask, _ = self.embed_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            proj_feat_2d=proj_feat_2d,
            labels=None
        )

        if self.config.model_specific.pred == "flow_matching":
            # TODO: not working after splitting action experts outside llm backbone
            raise NotImplementedError
            ret = None, self.generate_flow_matching(prefix_embeds, prefix_mask, prefix_block_mask, proprio, flow_matching_iter, llm_kwargs)[0]
        elif self.config.model_specific.pred in ["cot_flow_matching", "cot_bbox_flow_matching"]:
            # generate bbox and goal tokens autoregressively
            cot_parse, kv_cache = self.generate_autoregressive(
                input_ids=input_ids, 
                robot_input_ids=robot_input_ids,
                proj_feat_2d=proj_feat_2d,
                attention_mask=attention_mask, 
                robot_attention_mask=robot_attention_mask,
                max_token_num=max_token_num,
                token_pattern=token_pattern,
                inference_kwargs=inference_kwargs,
                require_kv_cache=True,
            )
            
            input_ids = torch.tensor(cot_parse.input_ids, device=input_ids.device)[None]
            _, prefix_mask, prefix_block_mask, _ = self.embed_prefix(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids).bool(),
                proj_feat_2d=proj_feat_2d,
                labels=None
            )
            
            padded_prefix_length = kv_cache[0][0].shape[2]
            num_paddings = padded_prefix_length - prefix_mask.shape[1]
            if num_paddings > 0:
                pad_mask = torch.zeros((prefix_mask.shape[0], num_paddings), dtype=prefix_mask.dtype, device=prefix_mask.device)
                prefix_mask = torch.cat([pad_mask, prefix_mask], dim=1)
                pad_block_mask = torch.zeros((num_paddings,), dtype=prefix_block_mask.dtype, device=prefix_block_mask.device)
                prefix_block_mask = torch.cat([pad_block_mask, prefix_block_mask], dim=0)
            
            # generate actions using flow matching
            action = self.generate_flow_matching(
                prefix_kv_cache=kv_cache,
                prefix_mask=prefix_mask, 
                prefix_block_mask=prefix_block_mask,
                proprio=proprio,
                flow_matching_iter=flow_matching_iter,
            )
            ret = cot_parse, action
        else:
            ret = self.generate_autoregressive(input_ids, robot_input_ids, proj_feat_2d, attention_mask, robot_attention_mask, max_token_num, token_pattern, inference_kwargs)[0], None
        return ret
    
    def prepare_flow_context(self, prefix_kv_cache, prefix_mask, prefix_block_mask, proprio):
        """Prepare the cached context shared by all flow integration steps."""
        device, dtype = prefix_kv_cache[0][0].device, prefix_kv_cache[0][0].dtype
        if not self.config.model_specific.action_expert:
            raise ValueError("stochastic flow sampling requires an action expert")
        if self.config.model_specific.llm.attn_implementation != "sdpa":
            raise ValueError("stochastic flow sampling currently requires SDPA")
        proprio_embeds = self.flow_module.proprior_proj(proprio.to(dtype))
        suffix_mask, suffix_block_mask = self.flow_module.get_suffix_masks(proprio_embeds)
        full_input_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        full_block_mask = torch.cat((prefix_block_mask, suffix_block_mask), axis=0)
        full_attn_mask = make_block_attn_mask(full_input_mask, full_block_mask).to(dtype)
        full_position_ids = torch.cumsum(full_input_mask, dim=1) - 1
        return (
            tuple(prefix_kv_cache),
            proprio_embeds,
            full_attn_mask[:, :, -suffix_mask.shape[1]:, ...],
            full_position_ids[:, -suffix_mask.shape[1]:],
        )

    def predict_flow_velocity(self, context, x_t: torch.Tensor, time_vec: torch.Tensor):
        prefix_kv_cache, proprio_embeds, suffix_attn_mask, suffix_position_ids = context
        suffix_embeds = self.flow_module.embed_suffix_flow_matching_embeds(
            proprio_embeds, x_t, time_vec
        )
        action_expert_output = self.action_expert(
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            inputs_embeds=suffix_embeds,
            past_key_values=prefix_kv_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        action_hidden_states = action_expert_output.hidden_states[-1][:, -self.config.action_steps:]
        return self.flow_module.get_v_t(action_hidden_states)

    def stochastic_generate_flow_matching(
        self, context, flow_matching_iter=5, sigma=0.01, generator=None
    ):
        """Sample a stochastic flow trajectory and return final action plus transitions."""
        prefix_kv_cache, proprio_embeds, _, _ = context
        del prefix_kv_cache, proprio_embeds
        device = context[1].device
        dtype = context[1].dtype
        x_t = self.flow_module.sample_noise(
            batch_size=context[1].shape[0], device=device, dtype=dtype
        )
        dt = -1.0 / flow_matching_iter
        transitions = []
        for step in range(flow_matching_iter):
            time_vec = torch.full((x_t.shape[0],), 1.0 + step * dt, device=device, dtype=dtype)
            v_t = self.predict_flow_velocity(context, x_t, time_vec)
            next_x, log_prob = self.flow_module.stochastic_euler_step(
                x_t, v_t, dt, sigma, generator=generator
            )
            transitions.append((x_t, next_x, time_vec, log_prob))
            x_t = next_x
        return x_t, transitions

    def generate_flow_matching(self, prefix_kv_cache, prefix_mask, prefix_block_mask, proprio, flow_matching_iter):
        context = self.prepare_flow_context(prefix_kv_cache, prefix_mask, prefix_block_mask, proprio)
        device, dtype = context[1].device, context[1].dtype
        noise = self.flow_module.sample_noise(batch_size=len(proprio), device=device, dtype=dtype)

        def compute_v_t(x_t: torch.Tensor, time_vec: torch.Tensor):
            return self.predict_flow_velocity(context, x_t, time_vec)

        return self.flow_module.denoise(compute_v_t, noise, flow_matching_iter)

    def generate_autoregressive(self, input_ids, robot_input_ids, proj_feat_2d, attention_mask, robot_attention_mask, max_token_num, token_pattern, inference_kwargs, require_kv_cache=False) -> Tuple[TokenPattern, Optional[Any]]:
        """Returns token pattern and kv cache.
        Requires batch size == 1 and no padding and no block attention.
        require_key_values enforces returning all_key_values in the cache.
        Note that this all_key_values includes things computed with the last token for flow matching, take care!
        """
        assert input_ids.shape[0] == 1, "only support single sample for now"
        cache = None
        current_input_embeddings = []
        current_input_mask = []
        current_block_mask = []
        pending = 0
        total_length = 0
        output = []
        for idx, token_info in enumerate([*token_pattern.infos, *token_pattern.robot_infos]):
            if token_info is None:
                continue
            if token_info.as_input:
                embeddings = self.llm.input_embedding(torch.tensor(inference_kwargs[0][token_info.key], device=input_ids.device))
                if idx == 0:
                    # insert the proj_feat_2d after the first embedding
                    embeddings = self.insert_img_info_single(embeddings, proj_feat_2d[0])
                current_input_embeddings.append(embeddings)
                current_block_mask.extend([0] * embeddings.shape[0])
                current_input_mask.extend([1] * embeddings.shape[0])
                pending += embeddings.shape[0]
                total_length += embeddings.shape[0]
                continue
            
            # let the network generate, then clear pending, and update kv cache

            generated_tokens, cache = self.llm.generate(
                max_token_num=token_info.length,
                inputs_embeds=torch.concat(current_input_embeddings, dim=0).unsqueeze(0),
                cache=cache,
            )
            total_length += len(generated_tokens[0])
            output.extend(generated_tokens[0])
            
            # reset pending tokens, it should be the embedding of the last generated token
            # assumes the kv cache does not contain the last token
            current_input_embeddings = [self.llm.input_embedding(torch.tensor(generated_tokens[0][-1:], dtype=torch.long, device=input_ids.device))]
            current_input_mask = [1]
            current_block_mask = [0]
            pending = 1
           
            # check completion
            parse_ret = token_pattern.update_tokens(output, **inference_kwargs[0])
            if parse_ret.terminate or len(output) >= max_token_num:
                break
        kv_cache = None
        if require_kv_cache and len(current_input_embeddings) != 0:
            _, cache_with_past_key_values = self.llm.generate(
                max_token_num=1,
                inputs_embeds=torch.concat(current_input_embeddings, dim=0).unsqueeze(0),
                cache=cache,
            )
            kv_cache = cache_with_past_key_values['past_key_values']
        return parse_ret, kv_cache
