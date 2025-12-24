"""Tests for LMHead and PLMModel (MLM-only protein language model)."""

import torch
import torch.nn as nn

from libreplm.models.libreplm import LMHead, PLMModel


class TestLMHead:
    """Tests for the LMHead class."""

    def test_lm_head_output_shape(self):
        """Test that LMHead produces correct output shape."""
        d_model = 64
        vocab_size = 32
        batch_size = 2
        seq_len = 16

        head = LMHead(d_model=d_model, vocab_size=vocab_size)
        x = torch.randn(batch_size, seq_len, d_model)

        output = head(x)

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert output.dtype == torch.float32

    def test_lm_head_with_weight_tying(self):
        """Test that LMHead correctly ties weights with embedding."""
        d_model = 64
        vocab_size = 32

        embed = nn.Embedding(vocab_size, d_model)
        head = LMHead(d_model=d_model, vocab_size=vocab_size, tie_weights=embed)

        # Check that decoder weight is the same object as embed weight
        assert head.decoder.weight is embed.weight

    def test_lm_head_without_weight_tying(self):
        """Test that LMHead creates independent weights without tying."""
        d_model = 64
        vocab_size = 32

        embed = nn.Embedding(vocab_size, d_model)
        head = LMHead(d_model=d_model, vocab_size=vocab_size, tie_weights=None)

        # Check that decoder weight is NOT the same object as embed weight
        assert head.decoder.weight is not embed.weight

    def test_lm_head_gradient_flow(self):
        """Test that gradients flow through LMHead."""
        d_model = 64
        vocab_size = 32

        head = LMHead(d_model=d_model, vocab_size=vocab_size)
        x = torch.randn(2, 8, d_model, requires_grad=True)

        output = head(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestPLMModel:
    """Tests for PLMModel (MLM-only protein language model)."""

    def test_model_creation(self):
        """Test that PLMModel can be created."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.1,
            attn_dropout=0.0,
            tie_word_embeddings=True,
        )

        assert model.lm_head is not None
        assert hasattr(model, "embed")
        assert hasattr(model, "encoder")

    def test_model_forward_pass(self):
        """Test forward pass through MLM model."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
        )
        model.eval()

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(4, 24, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(tokens=tokens)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 32)
        assert outputs["loss"] is None  # No labels provided

    def test_model_loss_computation(self):
        """Test that MLM model computes loss correctly with labels."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
        )

        batch_size = 2
        seq_len = 16
        tokens = torch.randint(4, 24, (batch_size, seq_len))

        # Create labels: only some positions are supervised
        labels = torch.full((batch_size, seq_len), -100)
        labels[:, 3:6] = tokens[:, 3:6]  # Supervise positions 3-5

        outputs = model(tokens=tokens, labels=labels, ignore_index=-100)

        assert outputs["loss"] is not None
        assert outputs["classification_loss"] is not None
        assert torch.isfinite(outputs["loss"])
        assert outputs["loss"].item() > 0

    def test_model_gradient_flow(self):
        """Test that gradients flow through MLM model during training."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
        )
        model.train()

        tokens = torch.randint(4, 24, (2, 16))
        labels = torch.full((2, 16), -100)
        labels[:, 3:6] = tokens[:, 3:6]

        outputs = model(tokens=tokens, labels=labels)
        loss = outputs["loss"]
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"

    def test_model_weight_tying(self):
        """Test that MLM model ties embedding weights when specified."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
            tie_word_embeddings=True,
        )

        # Check that decoder weight is tied to embedding weight
        assert model.lm_head.decoder.weight is model.embed.weight

    def test_model_no_weight_tying(self):
        """Test that MLM model does not tie weights when specified."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
            tie_word_embeddings=False,
        )

        # Check that decoder weight is NOT tied to embedding weight
        assert model.lm_head.decoder.weight is not model.embed.weight

    def test_model_attention_weights_output(self):
        """Test that model can return attention weights."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
        )
        model.eval()

        tokens = torch.randint(4, 24, (2, 16))

        with torch.no_grad():
            outputs = model(tokens=tokens, output_attentions=True)

        assert "attentions" in outputs
        # Should be a tuple of attention weights from each layer
        assert isinstance(outputs["attentions"], tuple)
        assert len(outputs["attentions"]) == 2  # n_layers = 2

    def test_model_hidden_states_output(self):
        """Test that model can return hidden states."""
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=2.667,
            dropout=0.0,
            attn_dropout=0.0,
        )
        model.eval()

        tokens = torch.randint(4, 24, (2, 16))

        with torch.no_grad():
            outputs = model(tokens=tokens, output_hidden_states=True)

        assert "hidden_states" in outputs
        # Should be a tuple of hidden states from each layer (including input embeddings)
        assert isinstance(outputs["hidden_states"], tuple)
        assert len(outputs["hidden_states"]) == 3  # n_layers + 1 (initial embeddings)
