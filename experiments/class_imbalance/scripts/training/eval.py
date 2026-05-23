from __future__ import annotations

import torch
from torch import nn

from scripts.mil.bags import AttentionMil, BagFeatureDataset, DualExpertMil, bag_collate
from scripts.training.support import _metric_payload


def _save_and_evaluate_bags(
    model: nn.Module,
    val_dataset: BagFeatureDataset,
    test_dataset: BagFeatureDataset,
    class_names: list[str],
    device: torch.device,
    result_dir,
) -> dict[str, dict[str, object]]:
    """Evaluate and save a feature-bag model."""
    val_results = _evaluate_bags(model, val_dataset, class_names, device)
    test_results = _evaluate_bags(model, test_dataset, class_names, device)
    torch.save(model.state_dict(), result_dir / "model.pt")
    return {"val": val_results, "test": test_results}


def _evaluate_bags(
    model: nn.Module,
    dataset: BagFeatureDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=bag_collate
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for bags, targets in loader:
            logits = _bag_logits(model, [bag.to(device) for bag in bags])
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def _bag_logits(model: nn.Module, bags: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(model, DualExpertMil):
        return model.forward_ensemble(bags)
    if isinstance(model, AttentionMil):
        return model.forward_bags(bags)[0]
    raise TypeError(f"Unsupported bag model: {type(model).__name__}")
