import torch
import torch.nn as nn

from .attention import MultiheadAttention
from .mlp import SwiGLU
from .norms import RMSNorm


class EncoderBlock(nn.Module):
    """Transformer encoder block with configurable pre/post normalization."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float,
        resid_dropout: float,
        rope: object,
        norm_type: str = "layernorm",
        ffn_mult: float = 4.0,
        pre_norm: bool = True,
        post_norm: bool = False,
        qk_norm: str = "none",
    ):
        """Initialize encoder block with configurable normalization.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention outputs.
            resid_dropout: Dropout probability for residual connections.
            rope: Rotary positional embedding instance.
            norm_type: Normalization type ("layernorm" or "rmsnorm").
            ffn_mult: Feedforward multiplier (hidden_dim = d_model * ffn_mult).
            pre_norm: Apply normalization before sublayer (default: True).
            post_norm: Apply normalization after residual (default: False).
            qk_norm: QK normalization type ("none", "norm", or "learned_scale").
        """
        super().__init__()

        # Validate at least one of pre_norm or post_norm must be True
        if not pre_norm and not post_norm:
            raise ValueError("At least one of pre_norm or post_norm must be True")

        self.pre_norm = pre_norm
        self.post_norm = post_norm

        # Norm class selection
        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            Norm = RMSNorm
        elif norm_type == "layernorm":
            Norm = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Pre-norm layers (applied BEFORE sublayer)
        self.norm1 = Norm(d_model) if pre_norm else None
        self.norm2 = Norm(d_model) if pre_norm else None

        # Post-norm layers (applied AFTER residual)
        self.post_norm1 = Norm(d_model) if post_norm else None
        self.post_norm2 = Norm(d_model) if post_norm else None

        self.attn = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attn_dropout,
            rope=rope,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.drop1 = nn.Dropout(resid_dropout)

        self.mlp = SwiGLU(d_model=d_model, expansion=ffn_mult, dropout=resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder block.

        Args:
            x: Input tensor of shape [B, L, d_model].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. Defaults to None.
            attn_mask: Attention mask of shape [B, H, L, S] or [B, 1, L, S].
                Defaults to None.
            output_attentions: If True, also returns attention weights.
                Defaults to False.

        Returns:
            If output_attentions=False: Output tensor of shape [B, L, d_model].
            If output_attentions=True: Tuple of (output, attention_weights) where
                attention_weights has shape [B, H, L, L].
        """
        # ----- Self-Attention Block -----
        # Pre-norm: h = norm(x) before sublayer
        h = self.norm1(x) if self.pre_norm else x

        attn_output = self.attn(
            h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=output_attentions,
        )

        if output_attentions:
            h, attn_weights = attn_output
        else:
            h = attn_output
            attn_weights = None

        # Residual connection
        x = x + self.drop1(h)

        # Post-norm: x = norm(x) after residual
        if self.post_norm:
            x = self.post_norm1(x)

        # ----- MLP Block -----
        # Pre-norm: h = norm(x) before sublayer
        h = self.norm2(x) if self.pre_norm else x
        h = self.mlp(h)

        # Residual connection
        x = x + self.drop2(h)

        # Post-norm: x = norm(x) after residual
        if self.post_norm:
            x = self.post_norm2(x)

        if output_attentions:
            return x, attn_weights
        return x
