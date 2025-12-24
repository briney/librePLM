import torch
import torch.nn.functional as F


def token_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy loss over tokens.

    Args:
        logits: Logits tensor of shape [B, L, C].
        labels: Target labels of shape [B, L].
        ignore_index: Index to ignore in loss computation. Defaults to -100.

    Returns:
        Scalar loss tensor.
    """
    C = int(logits.size(-1))
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    # Treat any label outside [0, C) as ignore_index to avoid device asserts
    invalid = (labels_flat < 0) | (labels_flat >= C)
    if invalid.any():
        labels_flat = labels_flat.clone()
        labels_flat[invalid] = ignore_index
    return F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
    )
