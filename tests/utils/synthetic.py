import random
from typing import Callable, List, Tuple

import torch

# Amino-acid-like alphabet consistent with Tokenizer's DEFAULT_VOCAB letters
_AMINO_ALPHABET = list("LAGV SERTIDPKQNFYMHW CXBUOZ.-".replace(" ", ""))


def random_protein_sequence(min_len: int = 50, max_len: int = 200) -> str:
    """Generate a random protein-like sequence.

    Args:
        min_len: Minimum sequence length. Defaults to 50.
        max_len: Maximum sequence length. Defaults to 200.

    Returns:
        Random protein sequence string.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choice(_AMINO_ALPHABET) for _ in range(length))


def build_mlm_batch(
    tokenizer, seqs: List[str], ignore_index: int, mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of tokenized sequences with MLM labels.

    Args:
        tokenizer: Tokenizer instance.
        seqs: List of protein sequence strings.
        ignore_index: Index to use for ignored labels.
        mask_prob: Probability of masking a token. Defaults to 0.15.

    Returns:
        Tuple of (input_ids, labels) with shapes [B, L] and [B, L].
    """
    enc = tokenizer(seqs, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]  # [B, L]
    attn = enc["attention_mask"]  # [B, L]
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    B, L = input_ids.shape
    labels = torch.full((B, L), fill_value=ignore_index, dtype=torch.long)

    # Valid positions = attended tokens that are not BOS/EOS
    valid_mask = (attn == 1) & (input_ids != bos_id) & (input_ids != eos_id)

    # Randomly select positions to mask
    mask_probs = torch.rand(B, L)
    mask_positions = valid_mask & (mask_probs < mask_prob)

    # Set labels at masked positions to the original token id
    labels[mask_positions] = input_ids[mask_positions]

    return input_ids.long(), labels.long()


def make_mlm_collate_fn(
    tokenizer, ignore_index: int, mask_prob: float = 0.15
) -> Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]:
    """Create a collate function for MLM tokenizing sequences.

    Args:
        tokenizer: Tokenizer instance.
        ignore_index: Index to use for ignored labels.
        mask_prob: Probability of masking a token. Defaults to 0.15.

    Returns:
        Collate function that takes a list of sequences and returns (tokens, labels).
    """

    def _collate(batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_mlm_batch(tokenizer, batch, ignore_index, mask_prob)

    return _collate
