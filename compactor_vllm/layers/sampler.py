import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        temps = temperatures.view(-1)
        scaled = logits.float()

        greedy_mask = temps == 0.0
        sample_mask = ~greedy_mask

        if sample_mask.any():
            temps_sample = temps[sample_mask].unsqueeze(-1)  # [B_sample, 1]
            scaled_sample = scaled[sample_mask].div(temps_sample)  # temperature scaling

            E = torch.empty_like(scaled_sample).exponential_(1).clamp_min_(1e-10).log()
            scaled_sample = scaled_sample - E

            scaled = scaled.clone()
            scaled[sample_mask] = scaled_sample

        return scaled.argmax(dim=-1)
