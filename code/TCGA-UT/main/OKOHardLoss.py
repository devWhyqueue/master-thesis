import torch
import torch.nn as nn
import torch.nn.functional as F

class OKOHardLoss(nn.Module):
    def __init__(self,):
        """
        """
        super().__init__()

    def forward(self, inputs, target):
        # inputs: (1, C) raw logits over the set dimension
        # target: (1,) set target

        target_indicator_vector = torch.zeros_like(inputs, dtype=inputs.dtype)
        target_indicator_vector[0, target] = 1

        log_probs = F.log_softmax(inputs, dim=1)
        loss = - target_indicator_vector.squeeze(0) @ log_probs.squeeze(0)
        return loss