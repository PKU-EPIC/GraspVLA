import torch
from torch import nn


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        self.projector = nn.Sequential(
            nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)
