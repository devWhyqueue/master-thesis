import torch
import torch.nn as nn
import torch.nn.functional as functional


class OKOHardLoss(nn.Module):
    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the odd-k-out hard-label set loss."""
        target_indicator = torch.zeros_like(inputs, dtype=inputs.dtype)
        target_indicator[0, target] = 1
        log_probs = functional.log_softmax(inputs, dim=1)
        return -target_indicator.squeeze(0) @ log_probs.squeeze(0)
