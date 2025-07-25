from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

from vla_network.type import BatchVLAData

from vla_network.config import VLADataConfig

    
def vla_collator(config: VLADataConfig, datas: List[BatchVLAData]) -> dict:
    kwargs = dict()

    pad_idx = config.tokenizer.pad_token_id
    max_len = config.tokenizer.model_max_length
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
    kwargs["input_ids"] = input_ids[:, :max_len]
    kwargs["robot_input_ids"] = robot_input_ids
    kwargs["attention_mask"] = kwargs["input_ids"] != pad_idx
    kwargs["robot_attention_mask"] = robot_input_ids != pad_idx

    for k in ['images', 'action', 'proprio', 'goal', 'is_action']:
        if getattr(datas[0], k, None) is not None:
            kwargs[k] = torch.cat([getattr(data, k) for data in datas], dim=0)
        else:
            kwargs[k] = None

    return kwargs