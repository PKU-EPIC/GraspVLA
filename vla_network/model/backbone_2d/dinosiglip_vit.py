import timm
import torch
from dataclasses import dataclass
from typing import List
from PIL import Image
from torchvision.transforms import Compose, Resize

from . import Backbone2D, Backbone2DConfig
from .common import ViT
from vla_network.config.define import ImageTransform

# Registry =>> Supported DinoSigLIP Pairs (as TIMM identifiers)
DINOSigLIP_NAMES = {
    224: {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    384: {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}


@dataclass
class CombineImageTransform:
    transforms: List[ImageTransform]

    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        return torch.stack([t(img, **kwargs) for t in self.transforms], dim=0)


class DinoSigLIPViTBackbone(Backbone2D):
    # from parent class
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    dino: ViT
    siglip: ViT

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__(config)
        self.models = ["dino", "siglip"]

        transforms = []
        for model_type in self.models:
            name = DINOSigLIP_NAMES[config.image_size][model_type]
            model: ViT = ViT(
                timm.create_model(
                    name, pretrained=True, num_classes=0, img_size=config.image_size
                )
            )
            model.eval()

            model_cfg = timm.data.resolve_model_data_config(model.model)
            model_cfg["input_size"] = (3, config.image_size, config.image_size)
            transform = timm.data.create_transform(**model_cfg, is_training=False)

            # Replace the resize transform with the target size
            target_size = (config.image_size, config.image_size)
            resize_transform = Compose(
                [
                    Resize(
                        target_size, interpolation=transform.transforms[0].interpolation
                    ),
                    *transform.transforms[1:],
                ]
            )

            setattr(self, model_type, model)
            transforms.append(resize_transform)
        self.image_transform = CombineImageTransform(transforms)

    @property
    def feature_dim(self) -> int:
        return self.dino.embed_dim + self.siglip.embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n, _, *chw = images.shape
        feats = []
        for i, k in enumerate(self.models):
            feat = getattr(self, k)(images[:, :, i].reshape(b * n, *chw))
            feats.append(feat.reshape(b, -1, feat.shape[-1]))
        return torch.cat(feats, dim=-1)

if __name__ == '__main__':
    for name in DINOSigLIP_NAMES[224].values():
        model: ViT = ViT(
            timm.create_model(
                name, pretrained=True, num_classes=0, img_size=224
            )
        )
