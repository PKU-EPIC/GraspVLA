from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import copy
import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedTokenizerBase, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import ModelOutput
import json
import re
from safetensors.torch import load_file
import os

from vla_network.type import RawVLAData
from vla_network.model.backbone_2d import Backbone2D
from vla_network.model.backbone_llm import LLMBackbone
from vla_network.config import VLAModelConfig, ImageTransform, VLADataConfig, ActionExpertConfig
from vla_network.data_preprocessing.preprocess import DataPreprocessor
from vla_network.data_preprocessing.vla_data_collator import vla_collator
from vla_network.data_preprocessing.token_pattern import TokenPattern, TokenResult
from vla_network.utils.constant import IGNORE_INDEX
from .projector import FusedMLPProjector
from .flow_matching import VLAFlowMatchingModule


def update_state_dict(state_dict: dict) -> dict:
    # update if load from prism vlm
    if "llm_backbone" in state_dict:
        state_dict["llm"] = state_dict.pop("llm_backbone")
    if "vision_backbone" in state_dict:
        state_dict["backbone_2d"] = dict()
        for k, v in state_dict.pop("vision_backbone").items():
            state_dict["backbone_2d"][k.replace("_featurizer", ".model")] = v
    return state_dict


def make_block_attn_mask(input_mask, block_mask):
    cumsum = torch.cumsum(block_mask, dim=0)
    causal_num = (cumsum == 0).sum()
    causal_mask = torch.tril(torch.ones((input_mask.shape[1], input_mask.shape[1]), dtype=torch.bool, device=input_mask.device))
    if causal_num != len(block_mask):
        block_attn_mask = cumsum[None, causal_num:] <= cumsum[causal_num:, None]
        causal_mask[causal_num:, causal_num:] = block_attn_mask
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(causal_mask, valid_mask)[:, None]


def load_safetensors(path: str) -> dict:
    return load_file(path)


def load_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = load_safetensors(ckpt_path)
    model.load_state_dict(ckpt)
    return model

class VLA(nn.Module, GenerationMixin):
    config: VLAModelConfig
    backbone_2d: Backbone2D
    llm: LLMBackbone
    projector: nn.Module
    train_modules: List[str]
    tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    is_train: bool

    def __init__(self, config: VLAModelConfig):
        super().__init__()
        self.config = config

    # TODO: Check whether add this to __init__ or not
    def init(self, train: bool = False):
        self.backbone_2d = Backbone2D.init(self.config.backbone_2d)
        self.backbone_2d_dim = self.backbone_2d.feature_dim
        self.image_transform = self.backbone_2d.image_transform
        self.is_train = train

        self.llm = LLMBackbone(self.config.llm, train=train)
        self.llm_dim = self.llm.input_dim
        self.tokenizer = self.llm.tokenizer

        if self.config.action_expert:
            self.action_expert = self.create_action_expert_from_llm(self.llm.llm, self.config.action_expert_cfg)

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(self.backbone_2d_dim)

        self.projector = FusedMLPProjector(self.backbone_2d_dim, self.llm_dim)
        
        if self.config.pred == "cot_flow_matching":
            self.flow_module = VLAFlowMatchingModule(
                config=self.config.flow_matching_cfg,
                action_dim=self.config.action_dim,
                llm_dim=self.action_expert.config.hidden_size,
                action_len=self.config.action_len,
                proprio_dim=self.config.proprio_dim,
            )

    @staticmethod
    def create_action_expert_from_llm(llm: PreTrainedModel, action_expert_config: ActionExpertConfig):
        config = copy.deepcopy(llm.config)
        if config.attn_implementation != "flex_attention":
            config.attn_implementation = "flex_attention"
        config.hidden_size = config.hidden_size // action_expert_config.hidden_size_scale
        config.intermediate_size = config.intermediate_size // action_expert_config.intermediate_size_scale
        config.hidden_act = config.hidden_act
        config.head_dim = llm.model.layers[0].attention.head_dim
        model_cls = type(llm)
        return model_cls._from_config(config)

    def from_pretrained(self, path: Optional[str] = None) -> "VLA":
        if path is None:
            path = self.config.ckpt
        state_dict = torch.load(path, map_location="cpu", weights_only=True)["model"]
        state_dict = update_state_dict(state_dict)
        if "backbone_2d" in state_dict:
            self.backbone_2d.load_state_dict(state_dict["backbone_2d"])
        self.projector.load_state_dict(state_dict["projector"])
        return self

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

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Optional[dict] = None
    ):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    # TODO: remove unused inputs
    # TODO: what should be the output type of this function?
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        robot_input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        robot_attention_mask: torch.Tensor = None,
        images: Optional[torch.FloatTensor] = None,
        proprio: Optional[torch.FloatTensor] = None,
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

        if self.config.pred == "cot_flow_matching":
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
            raise NotImplementedError(f"Prediction type {self.config.pred} is not implemented.")
        return ret
    
    def generate_flow_matching(self, prefix_kv_cache, prefix_mask, prefix_block_mask, proprio, flow_matching_iter):
        device, dtype = prefix_kv_cache[0][0].device, prefix_kv_cache[0][0].dtype
        assert self.config.action_expert
        assert self.config.llm.attn_implementation == "flex_attention"
        proprio = proprio.to(dtype)
        # TODO: should move to flow matching module instead of here
        noise = self.flow_module.sample_noise(
            batch_size=len(proprio),
            device=device,
            dtype=dtype
        )
        proprio_embeds = self.flow_module.proprior_proj(proprio)
        suffix_mask, suffix_block_mask = self.flow_module.get_suffix_masks(proprio_embeds)

        full_input_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        full_block_mask = torch.cat((prefix_block_mask, suffix_block_mask), axis=0)
        full_attn_mask = make_block_attn_mask(full_input_mask, full_block_mask).to(dtype)
        full_position_ids = torch.cumsum(full_input_mask, dim=1) - 1
        suffix_attn_mask = full_attn_mask[:, :, -suffix_mask.shape[1]:, ...]
        suffix_position_ids = full_position_ids[:, -suffix_mask.shape[1]:]

        prefix_kv_cache = tuple(prefix_kv_cache)

        def compute_v_t(x_t: torch.Tensor, time_vec: torch.Tensor):
            suffix_embeds = self.flow_module.embed_suffix_flow_matching_embeds(proprio_embeds, x_t, time_vec)
            action_expert_output = self.action_expert(
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                inputs_embeds=suffix_embeds,
                past_key_values=prefix_kv_cache, use_cache=True, output_hidden_states=True,
            )

            action_hidden_states = action_expert_output.hidden_states[-1][:, -self.config.action_len:]
            v_t = self.flow_module.get_v_t(action_hidden_states)
            return v_t

        x_0 = self.flow_module.denoise(compute_v_t, noise, flow_matching_iter)
        return x_0   

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


class VLAAgent():
    def __init__(self, path: Optional[str] = None, exp_name: Optional[str]=None, iter: Optional[int] = None, device: str = 'cuda:0', compile=False):
        self.path, self.exp_name, self.device, self.iter = path, exp_name, device, iter
        self.model_cfg, self.data_cfg, self.model, self.preprocessor = self.load_vla(path, exp_name, iter, device, compile)
        self.token_pattern = self.preprocessor.pattern

    def load_vla(
        self, path: Optional[str]=None, exp_name: Optional[str]=None, iter: Optional[int] = None, device: str = "cuda:0", compile=False,
    ) -> Tuple[VLAModelConfig, VLADataConfig, VLA, DataPreprocessor]:
        # TODO: return cfg as a VLAConfig
        cfg_path = os.path.join(os.path.dirname(path), '..', 'config.json')
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        data_cfg = VLADataConfig.model_validate(cfg["data"])
        model_cfg = VLAModelConfig.model_validate(cfg["model"])
        model: VLA = VLA(model_cfg)
        model.init(train=False)
        model = load_model(model, path)
        model = model.to(device).eval()
        if compile:
            model.llm.llm = torch.compile(model.llm.llm, dynamic=True)
            model.backbone_2d = torch.compile(model.backbone_2d)
            if hasattr(model, 'action_expert'):
                model.action_expert = torch.compile(model.action_expert, dynamic=True)
        data_cfg.tokenizer = model.tokenizer
        data_cfg.image_size = model.config.backbone_2d.image_size
        data_cfg.image_transform = model.image_transform
        data_cfg.pred = model_cfg.pred
        preprocessor = DataPreprocessor(data_cfg)
        preprocessor_path = os.path.join(os.path.dirname(path), '..', 'preprocessor.npz')
        preprocessor.load(np.load(preprocessor_path))
        return model_cfg, data_cfg, model, preprocessor

    def sample_action(self, raw: RawVLAData):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x = self.preprocessor.transform(raw, inference=True)
                model_input = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in vla_collator(self.data_cfg, [x]).items()}
                token_result, action_result = self.model.generate(
                    input_ids=model_input['input_ids'].to(self.device),
                    robot_input_ids=model_input['robot_input_ids'].to(self.device),
                    attention_mask=model_input['attention_mask'].to(self.device),
                    robot_attention_mask=model_input['robot_attention_mask'].to(self.device),
                    images=model_input['images'].to(self.device),
                    proprio=model_input['proprio'].to(self.device),
                    inference_kwargs=x.inference_kwargs,
                    token_pattern=self.token_pattern,
                    max_token_num=100,
                )
            ret = {}
            if self.model_cfg.pred == "cot_flow_matching":
                ret['action'] = self.preprocessor.robot_tokenizer.inv_norm_action(action_result.float().cpu().numpy()[0])
                if hasattr(token_result, 'goal'):
                    goal = self.preprocessor.robot_tokenizer.inv_goal(np.array(token_result.goal))
                    ret['goal'] = (goal[:3], goal[3:6])
                if hasattr(token_result, 'bbox'):
                    ret['bbox'] = (self.preprocessor.robot_tokenizer.uniform_tokenizer.uniform_detokenize(np.array(token_result.bbox).reshape(-1, 4)) + 1)/2*224
            else:
                raise NotImplementedError()
            return ret
    
    
    def __call__(self, samples: List) -> List[Dict[str, Any]]:
        rets = []
        for sample in samples:
            raw = RawVLAData(
                dataset_name="dummy",
                data_id=str(sample['env_id']),
                frame=0,
                instruction=sample['text'],
                images=dict(
                    front=np.stack(sample['front_view_image']),
                    side=np.stack(sample['side_view_image']),
                ),
                proprio=np.stack(sample['proprio_array']),
            )
            rets.append(self.sample_action(raw))
        return rets
