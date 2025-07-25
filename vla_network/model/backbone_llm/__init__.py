from typing import List, Optional

import torch
from torch import nn
import copy
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from vla_network.config import LLMConfig


PAD_TOKEN = "<PAD>"


class LLMBackbone(nn.Module):
    config: LLMConfig
    llm: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, config: LLMConfig, train: bool = False) -> None:
        super().__init__()

        config = copy.deepcopy(config)

        self.config = config
        
        self.llm = config.model_cls.from_pretrained(
            "internlm/internlm2-1_8b",
            attn_implementation=config.attn_implementation,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.llm.config.use_cache = True

        self.tokenizer = config.token_cls.from_pretrained(
            "internlm/internlm2-1_8b",
            model_max_length=self.config.max_len,
            padding_side="right",
            trust_remote_code=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.config.special_tokens}
        )
        self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(
            len(self.tokenizer), pad_to_multiple_of=config.pad_multiple_of
        )

    @property
    def input_dim(self) -> int:
        return self.input_embedding.embedding_dim

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the LLM given targets (labels), returning the scalar Cross-Entropy Loss"""
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def generate(
        self,
        max_token_num: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        requires_past_key_values: bool = False,
    ):
        """Contains optimization for generating a given number of tokens.
        NOTE: the returned cache should not contain things computed from the last generated token.
        """
        assert inputs_embeds.shape[0] == 1, "only single sample for now"
        return self.generate_normal(
            max_token_num,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )

    def generate_normal(
        self,
        max_token_num: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
        TODO: currently does not check termination, generates max_token_num for all sequences.
        """
        assert attention_mask is None
        assert position_ids is None
        if cache is None:
            cache = {}
        device = inputs_embeds.device if inputs_embeds is not None else next(self.llm.parameters()).device
        batch_size = inputs_embeds.shape[0] if inputs_embeds is not None else attention_mask.shape[0]
        generated_tokens = torch.zeros((batch_size, max_token_num), dtype=torch.long, device=device)
        past_key_values = cache.get("past_key_values")

        # construct attention mask and position ids, precompute for input_len + max_token_num so that we can simply slice the tensors during generation
        full_length = inputs_embeds.shape[1] + max_token_num
        full_length += past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_ids = torch.arange(full_length, device=device).unsqueeze(0)
        if self.llm.config.attn_implementation == "flex_attention":
            attention_mask = torch.tril(torch.ones((full_length, full_length), device=device)).unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = torch.ones((batch_size, full_length), device=device)

        # pad to multiples of PAD_TO to avoid torch recompile with varying seq len
        PAD_TO = 16
        num_padding = cache["num_padding"] if "num_padding" in cache else ((PAD_TO - (full_length % PAD_TO)) % PAD_TO)
        if num_padding > 0:
            # pad inputs_embeds only in prefill stage
            if past_key_values is None:
                pad_embeds = torch.zeros((inputs_embeds.shape[0], num_padding, inputs_embeds.shape[2]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = torch.cat([pad_embeds, inputs_embeds], dim=1)

            # pad position_ids
            pad_pos = torch.zeros((position_ids.shape[0], num_padding), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([pad_pos, position_ids], dim=1)

            # pad attention_mask
            if self.llm.config.attn_implementation == "flex_attention":
                pad_mask = torch.zeros((*attention_mask.shape[:2], num_padding, full_length), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([pad_mask, attention_mask], dim=2)
                pad_mask2 = torch.zeros((*attention_mask.shape[:2], full_length+num_padding, num_padding), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([pad_mask2, attention_mask], dim=3)
            else:
                # attention_mask: (batch_size, full_length)
                pad_mask = torch.zeros((attention_mask.shape[0], num_padding), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

        for i in range(max_token_num):
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask[:, :, past_length:past_length+inputs_embeds.shape[1], :past_length+inputs_embeds.shape[1]] if attention_mask is not None else None,
                position_ids=position_ids[:, past_length:past_length+inputs_embeds.shape[1]] if attention_mask is not None else None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1)
            generated_tokens[:, i] = next_tokens
            
            past_key_values = outputs.past_key_values
            inputs_embeds = self.llm.get_input_embeddings()(next_tokens.unsqueeze(-1))
        
        return generated_tokens.tolist(), {**cache, "past_key_values": past_key_values, "num_padding": num_padding}

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Optional[dict] = None
    ):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @property
    def input_embedding(self) -> nn.Embedding:
        return self.llm.get_input_embeddings()
