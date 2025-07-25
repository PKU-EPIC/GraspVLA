from copy import deepcopy
from typing import Callable, Dict, List
from tqdm import trange
import numpy as np

from vla_network.config import VLADataConfig

robot_tokenizer = None

class RobotTokenizer:
    config: VLADataConfig

    def __init__(self, config: VLADataConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size

    @staticmethod
    def init(config: VLADataConfig, vocab_size: int):
        global robot_tokenizer
        if robot_tokenizer is None:
            config = deepcopy(config)
            if config.tokenizer_type == "uniform":
                robot_tokenizer = UniformRobotTokenizer(config, vocab_size)
            elif config.tokenizer_type == "ratio_min_max_uniform":
                robot_tokenizer = RatioMinMaxUniformRobotTokenizer(config, vocab_size)
        return robot_tokenizer

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def goal(self, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self) -> dict:
        return {}


class UniformRobotTokenizer(RobotTokenizer):
    config: VLADataConfig
    bins: np.ndarray

    def __init__(self, config: VLADataConfig, vocab_size: int):
        super().__init__(config, vocab_size)
        self.bins = np.linspace(-1.0, 1.0, config.action_token_num)

    def uniform_tokenize(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        discretized_action = np.clip(np.digitize(x, self.bins), a_min=1, a_max=self.config.action_token_num)
        return self.vocab_size - discretized_action

    def uniform_detokenize(self, x: np.ndarray) -> np.ndarray:
        y = self.vocab_size - x
        return (
            self.bins[np.clip(y - 1, a_min=0, a_max=self.config.action_token_num - 1)]
            + self.bins[np.clip(y, a_min=0, a_max=self.config.action_token_num - 1)]
        ) / 2

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(bbox)

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(proprio)

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(action)

    def goal(self, goal: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(goal)

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        return self.uniform_detokenize(action)

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        return self.uniform_detokenize(goal)


class RatioMinMaxUniformRobotTokenizer(RobotTokenizer):
    config: VLADataConfig

    def __init__(self, config: VLADataConfig, vocab_size: int):
        super().__init__(config, vocab_size)
        self.uniform_tokenizer = UniformRobotTokenizer(config, vocab_size)

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        return self.uniform_tokenizer.bbox(bbox)

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        proprio = self.norm_proprio(proprio)
        return self.uniform_tokenizer.proprio(proprio)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = self.norm_action(action)
        return self.uniform_tokenizer.action(action)

    def goal(self, goal: np.ndarray) -> np.ndarray:
        goal = self.norm_goal(goal)
        return self.uniform_tokenizer.goal(goal)

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        action = self.uniform_tokenizer.inv_action(action)
        return self.inv_norm_action(action)

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        goal = self.uniform_tokenizer.inv_goal(goal)
        return self.inv_norm_goal(goal)

    def norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray):
        return (x - min_v) / (max_v - min_v) * 2 - 1

    def inv_norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray):
        return (x + 1) / 2 * (max_v - min_v) + min_v

    def norm_proprio(self, proprio: np.ndarray):
        return self.norm(proprio, self.min_proprio, self.max_proprio)

    def norm_action(self, action: np.ndarray):
        return self.norm(action, self.min_action, self.max_action)
    
    def norm_goal(self, goal: np.ndarray):
        return self.norm(goal, self.min_proprio[:-1], self.max_proprio[:-1])
    
    def inv_norm_action(self, action: np.ndarray):
        return self.inv_norm(action, self.min_action, self.max_action)

    def inv_norm_goal(self, goal: np.ndarray):
        return self.inv_norm(goal, self.min_proprio[:-1], self.max_proprio[:-1])

    def setup(self, get_func: Callable[[], Dict[str, np.ndarray]]):
        keys = list(get_func().keys())
        results = [[] for _ in keys]
        for _ in trange(self.config.count_num, desc="setup proprio action"):
            dic = get_func()
            for i in range(len(keys)):
                results[i].append(dic[keys[i]])
        for i in range(len(keys)):
            results[i] = np.stack(results[i])

        def set_min_max(data: np.ndarray, eps: float = 1e-7):
            data = data.reshape(-1, data.shape[-1])
            return (np.percentile(data, self.config.tokenizer_ratio_limit * 100, axis=0) - eps, 
                    np.percentile(data, (1 - self.config.tokenizer_ratio_limit) * 100, axis=0) + eps
            )

        self.min_proprio, self.max_proprio = set_min_max(results[keys.index("proprio")])
        self.min_action, self.max_action = set_min_max(results[keys.index("action")])

    def store_names(self) -> List[str]:
        ret = []
        for x in ["min", "max"]:
            for y in ["proprio", "action"]:
                ret.append(f"{x}_{y}")
        return ret

    def save(self) -> dict:
        ret = dict()
        for n in self.store_names():
            if getattr(self, n) is not None:
                ret[n] = getattr(self, n)
        return ret

    def load(self, data: dict):
        for n in self.store_names():
            if n in data:
                setattr(self, n, data[n])
            else:
                setattr(self, n, None)

