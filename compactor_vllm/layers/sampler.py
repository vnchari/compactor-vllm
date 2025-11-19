import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        scaled = logits.float().div_(temperatures.unsqueeze(dim=1))
        E = torch.empty_like(scaled).exponential_(1).clamp_min_(1e-10).log()
        return (scaled - E).argmax(dim=-1)
