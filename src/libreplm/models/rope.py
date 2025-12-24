import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache for RoPE.

    Builds cos/sin caches for a given maximum sequence length and head
    dimension, and applies them to query/key tensors.

    Attributes:
      base: RoPE base frequency.
      rope_dim: Dimensionality used for RoPE (must be even). If None, uses the
        last dimension of the head (head_dim).
    """

    def __init__(self, base: float = 10000.0, rope_dim: int | None = None):
        """Initialize rotary positional embedding.

        Args:
            base: RoPE base frequency. Defaults to 10000.0.
            rope_dim: Dimensionality used for RoPE (must be even). If None, uses
                the last dimension of the head (head_dim). Defaults to None.
        """
        super().__init__()
        self.base = base
        self.rope_dim = rope_dim
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self._cached_params: tuple[int, int] | None = None  # (seq_len, rope_dim)

    def _build_cache(
        self, seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Create cos/sin caches.

        Args:
          seq_len: Maximum sequence length to support.
          head_dim: Head dimension (D). RoPE will use rope_dim (<= D), must be even.
          device: Device for tensors.
          dtype: Floating dtype for caches.
        """
        rope_dim = self.rope_dim or head_dim
        assert rope_dim % 2 == 0, "rope_dim must be even"
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, rope_dim, 2, device=device, dtype=dtype) / rope_dim)
        )
        # Positions [0..L-1]
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("l,d->ld", t, inv_freq)  # [L, rope_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, rope_dim]
        self._cos = emb.cos()[None, None, :, :]  # [1,1,L,rope_dim]
        self._sin = emb.sin()[None, None, :, :]
        self._cached_params = (seq_len, rope_dim)

    def get_cos_sin(
        self, seq_len: int, head_dim: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) caches for a given seq_len and head_dim.

        Args:
            seq_len: Sequence length.
            head_dim: Head dimension.
            device: Device for tensors.
            dtype: Data type for tensors.

        Returns:
            Tuple of (cos, sin) caches with shape [1, 1, L, rope_dim].
        """
        if (
            (self._cos is None)
            or (self._cached_params is None)
            or (self._cached_params[0] < seq_len)
            or (self._cached_params[1] != (self.rope_dim or head_dim))
            or (self._cos.device != device)
            or (self._cos.dtype != dtype)
        ):
            self._build_cache(seq_len, head_dim, device, dtype)
        return self._cos[..., :seq_len, :], self._sin[..., :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension (even size) by splitting into halves.

    Args:
        x: Input tensor of shape [..., D] where D is even.

    Returns:
        Rotated tensor of shape [..., D] where the first half is -x[..., D/2:]
        and second half is x[..., :D/2].
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
      q: Query tensor of shape [B, H, L, D].
      k: Key tensor of shape [B, H, S, D].
      cos: Cosine cache of shape [1, 1, max(L,S), rope_dim], rope_dim <= D.
      sin: Sine cache of shape [1, 1, max(L,S), rope_dim].

    Returns:
      Tuple of (rotated_q, rotated_k) with shapes [B, H, L, D] and [B, H, S, D].
    """
    rope_dim = cos.size(-1)
    q1, q2 = q[..., :rope_dim], q[..., rope_dim:]
    k1, k2 = k[..., :rope_dim], k[..., rope_dim:]
    # Broadcast cos/sin to [B,H,L,rope_dim] and [B,H,S,rope_dim]
    q_cos = cos[..., : q.size(2), :]
    q_sin = sin[..., : q.size(2), :]
    k_cos = cos[..., : k.size(2), :]
    k_sin = sin[..., : k.size(2), :]
    q1 = q1 * q_cos + rotate_half(q1) * q_sin
    k1 = k1 * k_cos + rotate_half(k1) * k_sin
    return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)
