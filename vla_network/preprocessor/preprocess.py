from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np
from PIL import Image
import torch

from vla_network.config.define import VLAConfig
from vla_network.datasample import VLASample, BaseSample
from vla_network.utils.image import resize_image_with_bbox

from functools import singledispatchmethod
from collections import defaultdict, OrderedDict


class RatioMinMaxUniformRobotNormalizer:
    _instance: Any = None

    def __init__(self, num_samples, ratio):
        self.num_samples = num_samples
        self.ratio = ratio
        self.stats = {}

    @classmethod
    def init(cls, num_samples, ratio):
        if cls._instance is None:
            cls._instance = RatioMinMaxUniformRobotNormalizer(num_samples, ratio)
        return cls._instance

    def norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray, pad_to: Optional[int]):
        eps = 1e-7
        denominator = np.maximum(max_v - min_v, eps)
        normalized = (x - min_v) / denominator * 2 - 1
        normalized = np.clip(normalized, -1, 1)
        if pad_to is not None:
            current_size = normalized.shape[-1]
            assert pad_to >= current_size
            padded_x = np.zeros(normalized.shape[:-1] + (pad_to,), dtype=normalized.dtype)
            padded_x[..., :current_size] = normalized
            normalized = padded_x
        return normalized

    def inv_norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray, unpad: bool):
        if unpad:
            original_size = min_v.shape[-1]
            assert original_size <= x.shape[-1]
            x = x[..., :original_size]
        denormalized = (x + 1) / 2 * (max_v - min_v) + min_v
        return denormalized

    def norm_proprio(self, proprio: np.ndarray, embodiment: str, pad_to: Optional[int]):
        return self.norm(proprio, 
                         self.stats[embodiment]['proprio']['min'], 
                         self.stats[embodiment]['proprio']['max'],
                         pad_to)

    def norm_action(self, action: np.ndarray, embodiment: str, pad_to: Optional[int]):
        return self.norm(action, 
                         self.stats[embodiment]['action']['min'], 
                         self.stats[embodiment]['action']['max'],
                         pad_to)
    
    def norm_goal(self, goal: np.ndarray, embodiment: str, pad_to: Optional[int]):
        return self.norm(goal, 
                         self.stats[embodiment]['goal']['min'], 
                         self.stats[embodiment]['goal']['max'],
                         pad_to)
    
    def inv_norm_action(self, action: np.ndarray, embodiment: str, unpad: bool):
        return self.inv_norm(action, 
                            self.stats[embodiment]['action']['min'], 
                            self.stats[embodiment]['action']['max'],
                            unpad)

    def inv_norm_goal(self, goal: np.ndarray, embodiment: str, unpad: bool):
        return self.inv_norm(goal, 
                            self.stats[embodiment]['goal']['min'], 
                            self.stats[embodiment]['goal']['max'],
                            unpad)

    def setup(self, get_func: Callable[[], Dict[str, np.ndarray]]):
        embodiment_data = defaultdict(lambda: defaultdict(list))
        embodiment_counts = defaultdict(int)
        
        while True:
            all_complete = True
            for embodiment, count in embodiment_counts.items():
                if count < self.num_samples:
                    all_complete = False
                    break
            if all_complete and len(embodiment_data.keys()) >= 1:
                break
                
            embodiment, dic = get_func()
            if embodiment_counts[embodiment] >= self.num_samples:
                continue

            for key in dic:
                embodiment_data[embodiment][key].append(dic[key])
            embodiment_counts[embodiment] += 1
            
            # Progress reporting
            total_samples = sum(embodiment_counts.values())
            if total_samples % 100 == 0:
                embodiment_status = ", ".join([f"{e}: {c}/{self.num_samples}" 
                                             for e, c in embodiment_counts.items()])
                print(f"Collected {total_samples} samples. Status: {embodiment_status}")

        def set_min_max(data: np.ndarray):
            data = data.reshape(-1, data.shape[-1])
            return (np.percentile(data, self.ratio * 100, axis=0), 
                    np.percentile(data, (1 - self.ratio) * 100, axis=0)
            )
        
        for embodiment, data_dict in embodiment_data.items():
            if embodiment not in self.stats:
                self.stats[embodiment] = {}
            for key in data_dict:
                if key not in self.stats[embodiment]:
                    self.stats[embodiment][key] = {'min': None, 'max': None}
                
                data = np.stack(data_dict[key])
                data_min, data_max = set_min_max(data)
                self.stats[embodiment][key]['min'] = data_min
                self.stats[embodiment][key]['max'] = data_max

            print(f"Processed statistics for embodiment: {embodiment}")


class DataPreprocessor:
    config: VLAConfig
    robot_normalizer: RatioMinMaxUniformRobotNormalizer

    def __init__(self, config: VLAConfig):
        self.config = config
        self.robot_normalizer = RatioMinMaxUniformRobotNormalizer.init(
            config.train.normalizer_num_samples, 
            config.train.normalizer_ratio_limit
        )
        self.image_keys = {}

    def setup(self, load: Callable[[], VLASample]):
        def get_transform_input():
            while True:
                sample = load()
                if isinstance(sample, VLASample):
                    break
            sample: VLASample = self.process_input(sample, normalize=False)
            return sample.embodiment, {
                'proprio': sample.proprio, 
                'goal': sample.goal, 
                'action': sample.action
            }

        self.robot_normalizer.setup(get_transform_input)

    def process_input(self, sample: BaseSample, normalize: bool):
        """
        Currently resize images, unify invalid bbox representation, and normalize proprio & goal & action.
        """
        sample = self.transform(sample)
        if normalize:
            sample = self.normalize(sample)
        return sample

    @singledispatchmethod
    def transform(self, sample: BaseSample) -> BaseSample:
        raise NotImplementedError()
    @transform.register
    def _(self, sample: VLASample) -> VLASample:
        # check sample image key consistency
        if self.config.model.image_keys is None:
            if sample.embodiment not in self.image_keys:
                self.image_keys[sample.embodiment] = list(sample.images.keys())
            else:
                assert list(sample.images.keys()) == self.image_keys[sample.embodiment], f"inconsistent image keys for embodiment {sample.embodiment}"
        else:
            assert list(sample.images.keys()) == self.config.model.image_keys, "a VLA sample image keys differ from config image keys"

        images, bboxs = self._transform_img_bbox(sample.images, sample.bboxs)
        return sample.model_copy(update=dict(images=images, bboxs=bboxs))

    def _transform_img_bbox(self, raw_images: OrderedDict[str, List[Image.Image]], raw_bboxs: Optional[OrderedDict[str, np.ndarray]]) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """Resize image and bbox together, and make invalid bboxes as (0,0,0,0)."""
        if self.config.model.image_keys is not None:
            assert self.config.model.image_keys == list(raw_images.keys())
        images: OrderedDict[str, List[Image.Image]] = OrderedDict((k, []) for k in raw_images.keys())
        bboxs: OrderedDict[str, List[np.ndarray]] = OrderedDict((k, []) for k in raw_images.keys())

        assert all(len(raw_images[k]) == self.config.model.image_steps for k in raw_images.keys())
        for i in range(self.config.model.image_steps):
            for img_k in raw_images.keys():
                img, bbox = resize_image_with_bbox(
                    raw_images[img_k][i],
                    raw_bboxs[img_k][i] if raw_bboxs is not None else None,
                    (self.config.model.image_size, self.config.model.image_size),
                    random_padding=False,
                )
                if bbox is not None and (bbox[0] == bbox[2] or bbox[1] == bbox[3]):
                    bbox = np.zeros_like(bbox)
                images[img_k].append(img)
                bboxs[img_k].append(bbox)
        if raw_bboxs is None:
            bboxs = None
        return images, bboxs

    @singledispatchmethod
    def normalize(self, sample: BaseSample) -> BaseSample:
        return sample
    @normalize.register
    def _(self, sample: VLASample) -> VLASample:
        return sample.model_copy(update=dict(
            proprio=self.robot_normalizer.norm_proprio(sample.proprio, sample.embodiment, self.config.model.max_proprio_dim) if sample.proprio is not None else None,
            goal=self.robot_normalizer.norm_goal(sample.goal, sample.embodiment, self.config.model.max_goal_dim) if sample.goal is not None else None,
            action=self.robot_normalizer.norm_action(sample.action, sample.embodiment, self.config.model.max_action_dim) if sample.action is not None else None,
        ))

    def process_output(self, sample: BaseSample):
        sample = self.unnormalize(sample)
        return sample

    @singledispatchmethod
    def unnormalize(self, sample: BaseSample) -> BaseSample:
        raise NotImplementedError()
    @unnormalize.register
    def _(self, sample: VLASample) -> VLASample:
        return sample.model_copy(update=dict(
            goal=self.robot_normalizer.inv_norm_goal(sample.goal, sample.embodiment, unpad=True) if sample.goal is not None else None,
            action=self.robot_normalizer.inv_norm_action(sample.action, sample.embodiment, unpad=True) if sample.action is not None else None,
        ))
