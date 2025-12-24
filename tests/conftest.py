import sys
from pathlib import Path

import pytest
import torch

# Ensure the package under src/ is importable for tests
_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def device():
    """Pytest fixture providing a CPU device for tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tokenizer():
    """Pytest fixture providing a Tokenizer instance for tests."""
    # Ensure native deps exist; skip tests cleanly if not available
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from libreplm.utils.tokenizer import Tokenizer as _Tokenizer

    return _Tokenizer()


@pytest.fixture(scope="session")
def tiny_model_hparams(tokenizer):
    """Create model hyperparameters for MLM testing."""
    return dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
    )


@pytest.fixture(scope="session")
def ignore_index():
    """Pytest fixture providing the ignore index for loss computation."""
    return -100
