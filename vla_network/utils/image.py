import numpy as np
from typing import Tuple, Optional
from PIL import Image
import random


def resize_image_with_bbox(
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
