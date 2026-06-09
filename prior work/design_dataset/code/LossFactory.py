import torch
from FocalLoss import FocalLoss
from OKOHardLoss import OKOHardLoss

class LossFactory:

    @staticmethod
    def build(loss_type, gamma=None, alpha=None, n_classes=None, class_counts=None):

        if alpha == "uniform":
            if n_classes is None or n_classes < 0: ValueError(f"Invalid n_classes: {n_classes}")
            weights = torch.ones((n_classes), dtype=torch.double)
            weights = weights * (len(weights) / weights.sum()) # rescale to an average of 1 -> important to keep the loss on an appropriate scale
        elif alpha == "inverse_class_frequency":
            if class_counts is None or (class_counts < 0).any(): ValueError(f"Invalid class_counts: {class_counts}")
            weights = torch.tensor(1.0/class_counts)
            weights = weights * (len(weights) / weights.sum()) # rescale to an average of 1 -> important to keep the loss on an appropriate scale
        elif alpha is None:
            weights = None
        else:
            raise ValueError(f"Unknown alpha type: {alpha}")
        
        if loss_type == "cross_entropy":
            return torch.nn.CrossEntropyLoss(weight=weights)
        elif loss_type == "focal_loss":
            if gamma is None or gamma < 0:
                raise ValueError(f"Invalid gamma: {gamma}")
            loss = FocalLoss(gamma=gamma, alpha=weights)
            return loss
        elif loss_type == "oko_hard_loss":
            return OKOHardLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
