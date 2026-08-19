from copy import deepcopy
from typing import Callable, Dict, List
import numpy as np


class UniformTokenizer:
    bins: np.ndarray

    def __init__(self, token_num: int, vocab_size: int):
        self.token_num = token_num
        self.vocab_size = vocab_size
        self.bins = np.linspace(-1.0, 1.0, token_num)

    def uniform_tokenize(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        discretized_action = np.clip(np.digitize(x, self.bins), a_min=1, a_max=self.token_num)
        return self.vocab_size - discretized_action

    def uniform_detokenize(self, x: np.ndarray) -> np.ndarray:
        y = self.vocab_size - x
        return (
            self.bins[np.clip(y - 1, a_min=0, a_max=self.token_num - 1)]
            + self.bins[np.clip(y, a_min=0, a_max=self.token_num - 1)]
        ) / 2
