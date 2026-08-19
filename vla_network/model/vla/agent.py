from .base import BaseVLA, BaseSample

from vla_network.config import VLAConfig
from vla_network.preprocessor.preprocess import DataPreprocessor
from vla_network.model.vla import VLA
from vla_network.utils.ckpt import load_model
from vla_network.utils.path import (
    get_path_ckpt,
    get_path_exp,
    get_path_exp_config,
    get_path_exp_from_ckpt,
    get_path_preprocessor,
)
from vla_network.utils.logger import log

import json
import numpy as np
import torch
import pickle

from typing import Tuple, Optional, List, Dict, Any


class VLAAgent:
    def __init__(self, path: Optional[str] = None, exp_name: Optional[str]=None, iter: Optional[int] = None, device: str = 'cuda:0', compile=False):
        self.config, self.model, self.preprocessor = self.load_vla(path, exp_name, iter, device)
        if compile:
            self.model.compile()

    def load_vla(
        self, path: Optional[str]=None, exp_name: Optional[str]=None, iter: Optional[int] = None, device: str = "cuda:0"
    ) -> Tuple[VLAConfig, BaseVLA, DataPreprocessor]:
        if path is None:
            path = get_path_ckpt(get_path_exp(exp_name), iter)
        exp_path = get_path_exp_from_ckpt(path)
        with open(get_path_exp_config(exp_path=exp_path), "r", encoding="utf-8") as file:
            cfg = json.load(file)
        cfg['train']['args']['evaluation_strategy'] = 'no'
        cfg = VLAConfig.model_validate(cfg)
        model = VLA(cfg.model)
        model = load_model(model, path)
        if cfg.train.full_bf16:
            model = model.to(torch.bfloat16)
        model = model.to(device).eval()
        with open(get_path_preprocessor(exp_path=exp_path), "rb") as f:
            preprocessor = pickle.load(f)
        log.info("Model loaded")
        return cfg, model, preprocessor

    def __call__(self, samples: List[BaseSample]) -> List[BaseSample]:
        rets = []
        for sample in samples:
            sample = self.preprocessor.process_input(sample, normalize=True)

            # TODO: currently assumes whole bf16
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    sample = self.model.predict(sample)
            sample = self.preprocessor.process_output(sample)

            rets.append(sample)
        return rets
