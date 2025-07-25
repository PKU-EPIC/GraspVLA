from typing import Optional, Tuple, Union, Callable
import torch
from torch import nn

from vla_network.config import FlowMatchingConfig

def posemb_sincos(pos: torch.Tensor, embedding_dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=pos.dtype)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos, 1.0 / period * 2 * torch.pi)
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)

class BaseFlowMatchingModule(nn.Module):
    
    def __init__(self, config: FlowMatchingConfig, embed_dim: int):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        t_embed = posemb_sincos(t, self.embed_dim)
        return t_embed
    
    def sample_time(self, batch_shape: tuple, device: str, dtype: str) -> torch.FloatTensor:
        time = torch.distributions.Beta(
            torch.tensor(self.config.beta_alpha, device=device, dtype=torch.float32),
            torch.tensor(self.config.beta_beta, device=device, dtype=torch.float32)
        ).sample(batch_shape).to(dtype)
        time = time * (self.config.time_max - self.config.time_min) + self.config.time_min
        return time
        
    def sample_noise(self, shape: tuple, device: str, dtype: str) -> torch.FloatTensor:
        return torch.randn(shape, device=device, dtype=dtype)

    def diffuse(self, x_1: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_1)
        x_t = t * noise + (1 - t) * x_1
        u_t = noise - x_1
        return x_t, u_t
    

    def update(self, x_t: torch.FloatTensor, v_t: torch.FloatTensor, dt: Union[float, torch.FloatTensor], timestep: Union[float, torch.FloatTensor]) -> torch.FloatTensor:
        return x_t + dt * v_t
    
    def denoise(self, compute_v_t: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor], x_t: torch.FloatTensor, iter_num: int) -> torch.FloatTensor:
        device, dtype = x_t.device, x_t.dtype
        time_vec = torch.ones((len(x_t),), device=device, dtype=dtype)
        dt = 1.0 / iter_num
        time_steps = torch.linspace(1.0, dt, iter_num, device=device, dtype=dtype)
        for t in time_steps:
            time_vec[:] = t
            v_t = compute_v_t(x_t, time_vec)
            x_t = self.update(x_t, v_t, -dt, time_vec)
        return x_t

class VLAFlowMatchingModule(BaseFlowMatchingModule):    
    
    def __init__(self, config: FlowMatchingConfig, action_dim: int, llm_dim: int, action_len: int, proprio_dim: int):
        super().__init__(config=config, embed_dim=llm_dim)
        self.action_len = action_len
        self.action_dim = action_dim
        
        self.proprior_proj = nn.Linear(proprio_dim, llm_dim)
        self.action_in_proj = nn.Linear(action_dim, llm_dim)
        self.action_time_mlp = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim),
            nn.SiLU(),
            nn.Linear(llm_dim, llm_dim)
        )
        self.action_out_proj = nn.Linear(llm_dim, action_dim)
    
    def sample_noise_and_time(self, action: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_shape = action.shape[:-2]
        device, dtype = action.device, action.dtype
        
        time = self.sample_time(batch_shape, device=device, dtype=dtype)
        time_expanded = time[..., None, None]
        noise = super().sample_noise(action.shape, device=device, dtype=dtype)
        x_t, u_t = self.diffuse(action, time_expanded, noise)
        
        return x_t, u_t, time
    
    def sample_noise(self, batch_size: int, device: str, dtype: str) -> torch.FloatTensor:
        return super().sample_noise((batch_size, self.action_len, self.action_dim), device=device, dtype=dtype)
    
    def embed_suffix_flow_matching(
        self,
        proprio: torch.FloatTensor,
        noisy_actions: torch.FloatTensor,
        timestep: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]:
        #device = proprio.device
        dtype = self.proprior_proj.weight.dtype
        #batch_size = proprio.shape[0]
        
        proprio_embed = self.proprior_proj(proprio.to(dtype))
        embeds = self.embed_suffix_flow_matching_embeds(proprio_embed, noisy_actions, timestep)
        input_mask, block_mask = self.get_suffix_masks(proprio_embed)
        
        return embeds, input_mask, block_mask

    def get_suffix_masks(self, proprio_embed):
        batch_size = proprio_embed.shape[0]
        device = proprio_embed.device
        total_len = proprio_embed.shape[1] + self.action_len
        input_mask = torch.ones((batch_size, total_len), dtype=torch.bool, device=device)
        block_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        block_mask[0], block_mask[proprio_embed.shape[1]] = True, True
        return input_mask, block_mask

    def embed_suffix_flow_matching_embeds(
        self,
        proprio_embed: torch.FloatTensor,
        noisy_actions: torch.FloatTensor,
        timestep: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]:
        action_embeds = self.action_in_proj(noisy_actions)
        time_embeds = self.get_time_embedding(timestep)
        time_embeds = time_embeds[:, None, :].expand(-1, self.action_len, -1)
        action_time_embeds = self.action_time_mlp(
            torch.cat([action_embeds, time_embeds], dim=-1)
        )
        embeds = torch.cat([proprio_embed, action_time_embeds], dim=1)
        return embeds
    
    def get_v_t(self, hidden_states: torch.FloatTensor):
        return self.action_out_proj(hidden_states.to(self.action_out_proj.weight.dtype))
    

