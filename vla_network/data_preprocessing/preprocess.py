from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase
from transforms3d.euler import mat2euler, euler2mat

from vla_network.type import BatchVLAData, RawVLAData
from vla_network.config import VLADataConfig, ImageTransform

from .tokenizer import RobotTokenizer
from .token_pattern import get_token_pattern


def resize_with_bbox(
    image: Image.Image,
    bbox: Optional[np.ndarray],
    target_size: Tuple[int, int],
    random_padding: bool = True,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Resize the image to target size. Pad if necessary.
    Also computes the bbox on the resized & padded image.
    """
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGB", target_size)
    if random_padding:
        paste_x = random.randint(0, target_size[0] - new_size[0])
        paste_y = random.randint(0, target_size[1] - new_size[1])
    else:
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))

    if bbox is not None:
        new_bbox = bbox * ratio
        new_bbox[0] += paste_x
        new_bbox[1] += paste_y
        new_bbox[2] += paste_x
        new_bbox[3] += paste_y
        new_bbox = np.array([int(t) for t in new_bbox])
    else:
        new_bbox = None

    return new_image, new_bbox

class DataPreprocessor:
    config: VLADataConfig
    robot_tokenizer: RobotTokenizer
    tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform

    def __init__(self, config: VLADataConfig):
        self.config = config
        self.tokenizer = config.tokenizer
        config.tokenizer = None
        self.robot_tokenizer = RobotTokenizer.init(config, self.tokenizer.vocab_size)
        config.tokenizer = self.tokenizer
        self.image_transform = config.image_transform
        if config.pred == 'cot_flow_matching':
            self.pattern = get_token_pattern(config, 'cot_action')

    def load(self, data: dict):
        self.robot_tokenizer.load(data)

    def transform_img_bbox(self, raw_images: Dict[str, np.ndarray], raw_bboxs: Optional[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        pixel_values: List[Dict[str, torch.Tensor]] = []
        bboxs: List[np.ndarray] = []

        img_key = self.config.img_key
        assert all(len(raw_images[k]) == self.config.img_steps for k in img_key)
        for i in range(self.config.img_steps):
            for img_k in img_key:
                img, bbox = resize_with_bbox(
                    Image.fromarray(raw_images[img_k][i]),
                    raw_bboxs[img_k][i] if raw_bboxs is not None else None,
                    (self.config.image_size, self.config.image_size),
                )
                pixel_value = self.image_transform(img)
                if bbox is not None:
                    bbox = bbox / self.config.image_size * 2 - 1
                pixel_values.append(pixel_value)
                bboxs.append(bbox)
        pixel_values = torch.stack(pixel_values)[None]
        bboxs = np.stack(bboxs) if bboxs[0] is not None else None
        return pixel_values, bboxs


    def transform(self, raw_data: RawVLAData, inference: bool = False) -> BatchVLAData:

        pixel_values, bboxs = self.transform_img_bbox(raw_data.images, raw_data.bboxs)

        trans_dic = dict(proprio=raw_data.proprio, action=raw_data.action, goal=None)
        assert len(trans_dic["proprio"]) == self.config.proprio_len

        text_ids = self.tokenizer(raw_data.instruction, add_special_tokens=True).input_ids

        debug_dict = None
        inference_kwargs = [dict(
            text_ids=text_ids,
            hist_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][:-1]),
            cur_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][-1]),
        )]
        token_result = self.pattern.update_tokens(
            output=[], 
            **inference_kwargs[0]
        )
        input_ids = token_result.input_ids
        robot_input_ids = token_result.robot_input_ids

        return BatchVLAData(
            debug=[debug_dict],
            input_ids=torch.tensor(input_ids)[None],
            labels=None,
            attention_mask=torch.ones(len(input_ids))[None].bool(),
            robot_input_ids=torch.tensor(robot_input_ids)[None],
            robot_attention_mask=torch.ones(len(robot_input_ids))[None].bool(),
            robot_labels=None,
            images=pixel_values,
            action=None,
            proprio=torch.from_numpy(self.robot_tokenizer.norm_proprio(trans_dic['proprio'])).float()[None],
            goal=None,
            is_action=torch.ones(1).bool(),
            inference_kwargs=inference_kwargs,
        )