"""Comprehensive integration tests for P@L (Precision@L) metric.

These tests verify that the p_at_l metric is correctly built and computed
when coords are available during evaluation.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
from omegaconf import OmegaConf

from libreplm.cli.cli import cli
from libreplm.eval.registry import build_metrics, _get_dataset_has_coords
from libreplm.eval.metrics.contact import PrecisionAtLMetric, _compute_contact_map


class TestContactMapComputation:
    """Test the contact map computation logic."""

    def test_contact_map_basic(self):
        """Verify contact map computation works with valid coords."""
        B, L = 2, 20
        coords = torch.randn(B, L, 3, 3) * 5.0  # Random coords

        contact_map = _compute_contact_map(coords, threshold=8.0)

        # Check shape
        assert contact_map.shape == (B, L, L), (
            f"Contact map shape should be ({B}, {L}, {L})"
        )

        # Should be a boolean tensor
        assert contact_map.dtype == torch.bool, "Contact map should be boolean"

    def test_contact_map_with_nan_padding(self):
        """Verify contact map handles NaN padding correctly."""
        B, L = 2, 20
        coords = torch.randn(B, L, 3, 3) * 5.0

        # Add NaN padding for last 5 positions
        coords[:, 15:, :, :] = float("nan")

        contact_map = _compute_contact_map(coords, threshold=8.0)

        # Check shape
        assert contact_map.shape == (B, L, L), (
            f"Contact map shape should be ({B}, {L}, {L})"
        )

        # Valid region (positions 0-14) should not have NaN-related issues
        valid_region = contact_map[:, :15, :15]
        # These should be valid boolean values
        assert valid_region.dtype == torch.bool

        # NaN positions should be False (no contacts)
        # Check that NaN row/col positions are all False
        nan_row_contacts = contact_map[:, 15:, :15]  # NaN rows with valid cols
        nan_col_contacts = contact_map[:, :15, 15:]  # Valid rows with NaN cols
        assert not nan_row_contacts.any(), "NaN rows should have no contacts"
        assert not nan_col_contacts.any(), "NaN cols should have no contacts"


class TestPAtLMetricDirectly:
    """Test PrecisionAtLMetric directly with controlled inputs."""

    def test_p_at_l_metric_instantiation(self):
        """Verify P@L metric can be instantiated."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=6,
            use_attention=False,  # Force fallback path
        )
        assert metric.name == "p_at_l"
        assert metric.requires_coords is True
        assert metric.objectives == {"mlm"}

    def test_p_at_l_metric_update_without_exception(self):
        """Verify P@L metric update doesn't throw with valid inputs."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=3,  # Lower threshold for testing
            use_attention=True,
        )

        # Create mock inputs
        B, L, H = 2, 32, 4
        tokens = torch.randint(4, 24, (B, L))  # Amino acid tokens
        labels = torch.full((B, L), -100)  # All masked

        # Create realistic coords
        coords = torch.randn(B, L, 3, 3) * 10.0  # Scale to protein-like distances

        # Create mock outputs with attention weights (tuple of per-layer tensors)
        attentions = tuple(
            torch.softmax(torch.randn(B, H, L, L), dim=-1)
            for _ in range(2)  # 2 layers
        )

        outputs = {
            "logits": torch.randn(B, L, 32),  # vocab_size=32
            "loss": torch.tensor(2.5),
            "attentions": attentions,
        }

        # Create mock config
        cfg = OmegaConf.create(
            {
                "model": {
                    "encoder": {"pad_id": 1},
                }
            }
        )

        # This should not raise
        try:
            metric.update(outputs, tokens, labels, coords, cfg)
        except Exception as e:
            pytest.fail(f"metric.update raised an exception: {e}")

        # Compute should return valid result
        result = metric.compute()
        assert "p_at_l" in result, "Result should contain 'p_at_l' key"
        assert 0.0 <= result["p_at_l"] <= 1.0, "P@L should be between 0 and 1"

    def test_p_at_l_metric_with_attention_fallback(self):
        """Test P@L metric falls back to logits when no attention provided."""
        metric = PrecisionAtLMetric(
            contact_threshold=8.0,
            min_seq_sep=3,
            use_attention=True,  # Will try attention first
        )

        B, L = 2, 32
        tokens = torch.randint(4, 24, (B, L))
        labels = torch.full((B, L), -100)
        coords = torch.randn(B, L, 3, 3) * 10.0

        # No attention weights provided - should fall back to logits
        outputs = {
            "logits": torch.randn(B, L, 32),
            "loss": torch.tensor(2.5),
        }

        cfg = OmegaConf.create(
            {
                "model": {
                    "encoder": {"pad_id": 1},
                }
            }
        )

        # Should not raise even without attention
        metric.update(outputs, tokens, labels, coords, cfg)
        result = metric.compute()
        assert "p_at_l" in result


class TestMetricBuildingWithCoords:
    """Test that metrics are correctly built when coords are available."""

    def test_get_dataset_has_coords_explicit_load_coords(self, tmp_path):
        """Verify _get_dataset_has_coords returns True when load_coords is explicitly set."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "load_coords": True,
                        }
                    }
                }
            }
        )

        has_coords = _get_dataset_has_coords(cfg, "cameo", default_has_coords=False)
        assert has_coords is True, "Explicit load_coords=True should return True"

    def test_p_at_l_metric_built_for_mlm_with_coords(self, tmp_path):
        """Verify p_at_l metric is built when conditions are met."""
        cfg = OmegaConf.create(
            {
                "train": {"eval": {"metrics": {}}},  # No explicit disable
                "data": {
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "load_coords": True,
                        }
                    }
                },
            }
        )

        metrics = build_metrics(
            cfg=cfg,
            objective="mlm",
            has_coords=False,  # Default is False, but load_coords overrides
            eval_name="cameo",
        )

        metric_names = [m.name for m in metrics]
        assert "p_at_l" in metric_names, (
            f"p_at_l should be built for MLM with coords. Got metrics: {metric_names}"
        )


class TestEvaluatorAttentionPropagation:
    """Test that attention weights flow correctly through the evaluator."""

    def test_evaluator_needs_attentions_detection(self, tmp_path):
        """Test that evaluator detects when attention weights are needed."""
        from libreplm.eval.evaluator import Evaluator
        from libreplm.models.libreplm import PLMModel

        cfg = OmegaConf.create(
            {
                "train": {
                    "eval": {"metrics": {"p_at_l": {"enabled": True}}},
                },
                "data": {
                    "load_coords": False,
                    "max_len": 128,
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "load_coords": True,  # Explicitly enable coords
                        }
                    },
                },
                "model": {
                    "encoder": {
                        "vocab_size": 32,
                        "pad_id": 1,
                        "d_model": 64,
                        "n_heads": 4,
                        "n_layers": 2,
                        "ffn_mult": 1.0,
                        "dropout": 0.0,
                        "attn_dropout": 0.0,
                    },
                },
            }
        )

        # Create model
        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ffn_mult=1.0,
            dropout=0.0,
            attn_dropout=0.0,
        )

        evaluator = Evaluator(
            cfg=cfg,
            model=model,
            accelerator=None,
        )

        # Check that metrics were built correctly
        metrics = evaluator._get_metrics("cameo")
        metric_names = [m.name for m in metrics]

        assert "p_at_l" in metric_names, (
            f"p_at_l metric should be built. Got: {metric_names}"
        )

        # Check that needs_attentions is True for cameo dataset
        assert evaluator._needs_attentions("cameo") is True, (
            "Evaluator should detect that cameo dataset needs attention weights"
        )

    def test_evaluator_p_at_l_num_layers_config(self, tmp_path):
        """Test that num_layers config is correctly passed to P@L metric."""
        from libreplm.eval.evaluator import Evaluator
        from libreplm.models.libreplm import PLMModel

        cfg = OmegaConf.create(
            {
                "train": {
                    "eval": {
                        "metrics": {
                            "p_at_l": {
                                "enabled": True,
                                "num_layers": 3,  # Use 3 final layers
                            }
                        }
                    },
                },
                "data": {
                    "load_coords": False,
                    "max_len": 128,
                    "eval": {
                        "cameo": {
                            "path": str(tmp_path),
                            "load_coords": True,  # Explicitly enable coords
                        }
                    },
                },
                "model": {
                    "encoder": {
                        "vocab_size": 32,
                        "pad_id": 1,
                        "d_model": 64,
                        "n_heads": 4,
                        "n_layers": 6,  # 6 layers to test num_layers=3
                        "ffn_mult": 1.0,
                        "dropout": 0.0,
                        "attn_dropout": 0.0,
                    },
                },
            }
        )

        model = PLMModel(
            vocab_size=32,
            pad_id=1,
            d_model=64,
            n_heads=4,
            n_layers=6,
            ffn_mult=1.0,
            dropout=0.0,
            attn_dropout=0.0,
        )

        evaluator = Evaluator(
            cfg=cfg,
            model=model,
            accelerator=None,
        )

        # Check that metrics were built with correct num_layers
        metrics = evaluator._get_metrics("cameo")
        p_at_l = next(
            (m for m in metrics if m.name == "p_at_l"),
            None,
        )

        assert p_at_l is not None, "P@L metric should be built"
        assert p_at_l.num_layers == 3, (
            f"P@L num_layers should be 3, got {p_at_l.num_layers}"
        )


class TestEndToEndMLMWithCameoEval:
    """End-to-end integration tests for P@L metric behavior."""

    def test_mlm_training_p_at_l_disabled_by_default_no_coords(self, tmp_path):
        """
        Verify p_at_l is NOT computed when coords are not available.
        This ensures the metric correctly respects its requirements.
        """
        # Create CSV dataset without coords
        train_csv = tmp_path / "train.csv"
        eval_csv = tmp_path / "eval.csv"

        df = pd.DataFrame(
            {
                "pid": [f"train_{i}" for i in range(10)],
                "protein_sequence": [
                    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"[:30] for _ in range(10)
                ],
            }
        )
        df.to_csv(train_csv, index=False)

        eval_df = pd.DataFrame(
            {
                "pid": [f"eval_{i}" for i in range(5)],
                "protein_sequence": [
                    "MNIFEMLRIDKGLQVVAVKAPGFGDNRKNQ"[:25] for _ in range(5)
                ],
            }
        )
        eval_df.to_csv(eval_csv, index=False)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=64",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            f"+data.eval.validation={eval_csv.as_posix()}",
            # Don't enable p_at_l explicitly - it requires coords
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "eval/validation" in result.output

        # P@L should NOT be logged because there are no coords
        assert "P@L" not in result.output, (
            "P@L should not be logged when coords are not available"
        )
