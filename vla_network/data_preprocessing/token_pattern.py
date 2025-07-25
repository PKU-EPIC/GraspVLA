from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import torch
import numpy as np
from vla_network.utils.constant import IGNORE_INDEX
from vla_network.config import VLADataConfig

UNFINISHED = "unfinished_"

def to_flatten_list(x: Union["torch.Tensor", np.ndarray, list]) -> list:  # type: ignore
    """
    Convert a tensor, numpy array, or list to a flattened list.

    Parameters
    ----------
    x : Union["torch.Tensor", np.ndarray, list]
        The input data to convert.

    Returns
    -------
    list
        The flattened list.
    """
    if torch is not None and isinstance(x, torch.Tensor):
        return x.reshape(-1).tolist()
    elif isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    elif isinstance(x, list):
        return np.array(x).reshape(-1).tolist()
    else:
        raise ValueError(f"Unsupported type {type(x)}")

class TokenInfo(BaseModel):
    key: str # the key in the input dict
    length: Optional[int] # the length of the token, None means no limit
    est: bool # whether loss is calculated for those tokens
    as_input: bool # whether those tokens are input tokens
    # terminate function to determine if the token sequence is complete
    # the default is to check if the length of the tokens is equal to the length of the token info
    terminate: Callable[["TokenInfo", List[int]], bool] = Field(default=lambda tinfo, tokens: len(tokens) == tinfo.length)

    def model_post_init(self, __context):
        assert self.key not in ['terminate', 'input_ids', 'robot_input_ids'], f"key {self.key} is not allowed in TokenInfo"
        assert not self.key.startswith(UNFINISHED), f"key {self.key} is not allowed in TokenInfo"

@dataclass
class TokenResult:
    # whether the token sequence is complete
    terminate: bool = field(default_factory=lambda: False)
    # the new input ids of the tokens
    input_ids: List[int] = field(default_factory=lambda: [])
    # the new robot input ids of the tokens
    robot_input_ids: List[int] = field(default_factory=lambda: [])
    # allow arbitrary key(str): value(List[int]) 
    # note that there are two requirements of the keys (see TokenInfo.model_post_init):
    # 1. they should not conflict with the above keys
    # 2. they should not start with "unfinished_" since we use this prefix to indicate that the token is not finished

    # things not listed above is also shown in this func
    def __str__(self) -> str:
        ret = 'TokenResult:\n'
        for k, v in self.__dict__.items():
            ret += f'\t{k}={v}\n'
        return ret
    
    def __repr__(self) -> str:
        return self.__str__()

class TokenPattern(BaseModel):
    # the token info for the input tokens, N
    infos: List[Optional[TokenInfo]]
    # the token info for the robot input tokens
    robot_infos: List[Optional[TokenInfo]]

    # get the input ids and labels for the input tokens
    def get_input_id_label(self, **kwargs: Dict[str, List[int]])-> Tuple[List[int], List[int]]:
        return self.get_id_label_inner(self.infos, **kwargs)
    
    # get the input ids and labels for the robot input tokens
    def get_robot_input_id_label(self, **kwargs: Dict[str, List[int]]) -> Tuple[List[int], List[int]]:
        return self.get_id_label_inner(self.robot_infos, **kwargs)

    @staticmethod
    def get_id_label_inner(infos: List[TokenInfo], **kwargs: Dict[str, List[int]]) -> Tuple[List[int], List[int]]:
        input_ids, labels = [], []
        for info in infos:
            if info is None:
                continue
            value = to_flatten_list(kwargs.get(info.key, []))
            assert info.length is None or len(value) == info.length, f"key {info.key} length {len(value)} != {info.length}"
            # add to input ids
            input_ids.extend(to_flatten_list(value))
            if info.est:
                # add to labels
                labels.extend(to_flatten_list(value))
            else:
                # ignore those tokens's loss
                labels.extend([IGNORE_INDEX] * len(value))
        return input_ids, labels
    
    # suppose we output some of the tokens (maybe unfinished), we need to update the input ids and labels
    # can be used in generation
    def update_tokens(self, output: List[int], **kwargs: Dict[str, List[int]]) -> TokenResult:
        output = deepcopy(to_flatten_list(output))
        ret = TokenResult(terminate=False)
        # shallow copy the input ids so that we can add tokens to it
        for ids, infos in [(ret.input_ids, self.infos), (ret.robot_input_ids, self.robot_infos)]:
            for info in infos:
                if info is None:
                    continue

                if info.as_input:
                    # if the token is as_input, then we should find them in input and add them to ids
                    if info.key in kwargs:
                        value = to_flatten_list(kwargs[info.key])
                        ids.extend(value)
                    else:
                        value = None
                    setattr(ret, info.key, value)
                else:
                    # the token is not as_input, then we should find the tokens in the output
                    cur = []
                    while True:
                        if info.terminate(info, cur):
                            # if this part finished, then we should save to ret and break
                            setattr(ret, info.key, cur)
                            break
                        elif len(output) == 0:
                            # if the output is empty, then the rest of tokens haven't been predicted
                            # save the current tokens and return
                            # terminate is False
                            setattr(ret, UNFINISHED+info.key, cur)
                            return ret
                        else:
                            # the next token is one of the output tokens
                            token_id = output.pop(0)
                            ids.append(token_id)
                            cur.append(token_id)

        # all the tokens are predicted
        assert len(output) == 0, f"output is not empty, {output}"
        ret.terminate = True
        return ret
# fmt: off

def get_cot_action_pattern(config: VLADataConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key='text_ids', length=None, est=False, as_input=True),
            TokenInfo(key='bbox', length=config.img_num * 4, est=True, as_input=False) if config.use_bbox else None,
            TokenInfo(key='hist_proprio', length=(config.proprio_len-1) * config.proprio_dim, est=False, as_input=True),
            TokenInfo(key='cur_proprio', length=config.proprio_dim, est=False, as_input=True),
            TokenInfo(key='goal', length=config.goal_dim, est=True, as_input=False) if config.goal_dim is not None else None,
            TokenInfo(key='eos', length=1, est=True, as_input=False),
        ],
        robot_infos=[
        ],
    )


def get_token_pattern(config: VLADataConfig, name: str) -> TokenPattern:
    return dict(
        cot_action=get_cot_action_pattern,
    )[name](config)