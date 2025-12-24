import sys

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from libreplm.models.libreplm import PLMModel


def run_smoke_test(cfg: DictConfig):
    """Build the model and run a tiny forward pass for smoke testing.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print(OmegaConf.to_yaml(cfg))

    model = PLMModel(
        vocab_size=cfg.model.encoder.vocab_size,
        pad_id=cfg.model.encoder.pad_id,
        d_model=cfg.model.encoder.d_model,
        n_heads=cfg.model.encoder.n_heads,
        n_layers=cfg.model.encoder.n_layers,
        ffn_mult=cfg.model.encoder.ffn_mult,
        dropout=cfg.model.encoder.dropout,
        attn_dropout=cfg.model.encoder.attn_dropout,
        norm_type=cfg.model.encoder.norm,
        tie_word_embeddings=cfg.model.encoder.get("tie_word_embeddings", True),
    )

    if cfg.print_model_summary:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params/1e6:.2f}M")

    # Tiny forward sanity check for MLM
    B, L = 2, 16
    vocab_size = cfg.model.encoder.vocab_size
    pad_id = cfg.model.encoder.pad_id
    mask_id = cfg.model.encoder.get("mask_id", 31)  # default mask token ID (<mask> at index 31)
    ignore_index = cfg.train.mlm.get("ignore_index", -100)

    # Create input tokens (random amino acid tokens)
    tokens = torch.randint(low=1, high=vocab_size, size=(B, L))
    tokens[:, -2:] = pad_id  # Last 2 positions are padding

    # Create masked positions for MLM
    # Mask ~15% of tokens (2-3 positions out of 14 non-pad tokens)
    mask_positions = torch.zeros(B, L, dtype=torch.bool)
    mask_positions[:, 2] = True  # Mask position 2
    mask_positions[:, 5] = True  # Mask position 5

    # Create labels: -100 (ignore) for non-masked positions, original token for masked
    labels = torch.full((B, L), ignore_index, dtype=torch.long)
    labels[mask_positions] = tokens[mask_positions]

    # Replace masked positions with mask token in input
    tokens[mask_positions] = mask_id

    out = model(tokens=tokens, labels=labels, ignore_index=ignore_index)
    print("logits:", out["logits"].shape, "loss:", float(out["loss"].item()))
    print("OK")


if __name__ == "__main__":
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=overrides)
        run_smoke_test(cfg)
