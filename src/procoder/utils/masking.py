import torch


def key_padding_mask_from_tokens(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build key padding mask from token IDs.

    Args:
        tokens: Token IDs tensor of shape [B, L].
        pad_id: Padding token ID.

    Returns:
        Boolean mask of shape [B, L] where True marks padding positions.
    """
    return tokens == pad_id
