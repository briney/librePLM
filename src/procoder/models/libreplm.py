import torch
import torch.nn as nn

from ..utils.losses import token_ce_loss
from .encoder import Encoder


class LMHead(nn.Module):
    """Simple language modeling head for MLM pre-training."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: nn.Embedding | None = None,
    ):
        """Initialize LM head.

        Args:
            d_model: Model dimension.
            vocab_size: Vocabulary size for output logits.
            tie_weights: Optional embedding to tie weights with. If provided,
                the decoder weights will be shared with the embedding weights.
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

        if tie_weights is not None:
            # Weight tying with input embeddings
            self.decoder = nn.Linear(d_model, vocab_size, bias=False)
            self.decoder.weight = tie_weights.weight
        else:
            self.decoder = nn.Linear(d_model, vocab_size)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through LM head.

        Args:
            h: Hidden states of shape [B, L, d_model].

        Returns:
            Logits tensor of shape [B, L, vocab_size].
        """
        h = self.dense(h)
        h = self.activation(h)
        h = self.layer_norm(h)
        return self.decoder(h) + self.bias


class PLMModel(nn.Module):
    """Encoder-only protein language model for MLM pre-training.

    This model consists of a transformer encoder with a language modeling head
    for masked token prediction. It is designed for pre-training on protein
    sequences using the masked language modeling (MLM) objective.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: float,
        dropout: float,
        attn_dropout: float,
        norm_type: str = "layernorm",
        tie_word_embeddings: bool = True,
    ):
        """Initialize PLM model.

        Args:
            vocab_size: Vocabulary size for input tokens.
            pad_id: Padding token ID.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            ffn_mult: Feedforward multiplier (hidden_dim = d_model * ffn_mult).
            dropout: Dropout probability for residual connections.
            attn_dropout: Dropout probability for attention outputs.
            norm_type: Normalization type ("layernorm" currently supported).
            tie_word_embeddings: Whether to tie LM head weights to input embeddings.
        """
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_mult=ffn_mult,
            norm_type=norm_type,
        )

        self.lm_head = LMHead(
            d_model=d_model,
            vocab_size=vocab_size,
            tie_weights=self.embed if tie_word_embeddings else None,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through PLM model.

        Args:
            tokens: Input token IDs of shape [B, L].
            key_padding_mask: Padding mask of shape [B, L] where True marks
                padding positions. If None, inferred from pad_id. Defaults to None.
            labels: Target labels of shape [B, L]. Use ignore_index for
                ignored positions. Defaults to None.
            ignore_index: Index to ignore in loss computation. Defaults to -100.
            output_attentions: If True, also returns attention weights from all
                encoder layers. Defaults to False.
            output_hidden_states: If True, also returns hidden states from all
                encoder layers (including initial embeddings). Defaults to False.

        Returns:
            Dictionary containing:
                - logits: Output logits of shape [B, L, vocab_size].
                - loss: Cross-entropy loss if labels provided, else None.
                - classification_loss: Same as loss (for compatibility).
                - attentions: Tuple of attention weights from each encoder layer,
                    each of shape [B, H, L, L]. Only present if output_attentions=True.
                - hidden_states: Tuple of hidden states from each encoder layer
                    (including initial embeddings), each of shape [B, L, d_model].
                    Only present if output_hidden_states=True.
        """
        # embedding
        h = self.embed(tokens)  # [B, L, d_model]

        # if mask not provided, build from pad_id
        if key_padding_mask is None:
            key_padding_mask = tokens == self.pad_id  # [B, L], True = pad

        # encode
        encoder_output = self.encoder(
            h,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Unpack encoder output based on flags
        all_attentions = None
        all_hidden_states = None

        if output_attentions and output_hidden_states:
            h, all_attentions, all_hidden_states = encoder_output
        elif output_attentions:
            h, all_attentions = encoder_output
        elif output_hidden_states:
            h, all_hidden_states = encoder_output
        else:
            h = encoder_output

        # language modeling head
        logits = self.lm_head(h)  # [B, L, vocab_size]

        loss = None
        classification_loss = None

        # MLM loss
        if labels is not None:
            classification_loss = token_ce_loss(
                logits=logits,
                labels=labels,
                ignore_index=ignore_index,
            )
            loss = classification_loss

        result = {
            "logits": logits,
            "loss": loss,
            "classification_loss": classification_loss,
        }

        if output_attentions:
            result["attentions"] = all_attentions

        if output_hidden_states:
            result["hidden_states"] = all_hidden_states

        return result


# Backward compatibility alias
STokModel = PLMModel
