import torch


def top_p_sample(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    # logits: [B, L, C] -> sampled indices: [B, L]
    if temperature != 1.0:
        logits = logits / temperature
    B, L, C = logits.shape
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    to_remove = cumprobs > p
    to_remove[..., 0] = False  # keep at least the top-1
    sorted_probs = torch.where(to_remove, torch.zeros_like(sorted_probs), sorted_probs)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    ranks = torch.multinomial(sorted_probs.reshape(-1, C), 1).view(B, L, 1)
    sampled = torch.gather(sorted_idx, -1, ranks).squeeze(-1)
    return sampled
