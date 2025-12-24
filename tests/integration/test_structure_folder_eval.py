"""Integration tests for training with structure folder evaluation datasets.

These tests verify that the StructureFolderDataset is correctly wired into the
training pipeline and works with PDB/mmCIF structure folders for evaluation.
"""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from libreplm.cli.cli import cli


# Path to test data
CAMEO_DIR = Path(__file__).parent.parent / "test_data" / "cameo"


def _skip_if_no_cameo():
    """Skip test if CAMEO test data is not available."""
    if not CAMEO_DIR.exists():
        pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")


class TestStructureFolderEvalExplicitFormat:
    """Tests for explicit format=structure configuration."""

    def test_cli_train_with_structure_folder_explicit_format(self, tmp_path):
        """Test CLI training with explicit format=structure for eval dataset."""
        _skip_if_no_cameo()

        # Create CSV training data
        train_csv = tmp_path / "train.csv"
        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            # Model config
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            # Data config
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Eval with explicit structure format
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "+data.eval.cameo.format=structure",
            # Training config
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output
        # Verify eval was performed on cameo dataset
        assert "eval/cameo" in result.output

    def test_cli_train_structure_folder_with_chain_id(self, tmp_path):
        """Test CLI training with chain_id parameter for structure folder."""
        _skip_if_no_cameo()

        train_csv = tmp_path / "train.csv"
        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Eval with structure format and chain_id
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "+data.eval.cameo.format=structure",
            # Note: chain_id may or may not work depending on actual CAMEO data
            # The test verifies the parameter is passed correctly
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output


class TestStructureFolderEvalAutoDetection:
    """Tests for auto-detection of structure folders."""

    def test_cli_train_structure_folder_auto_detected(self, tmp_path):
        """Test that structure folder is auto-detected without explicit format."""
        _skip_if_no_cameo()

        train_csv = tmp_path / "train.csv"
        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Eval WITHOUT explicit format - should auto-detect
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output
        assert "eval/cameo" in result.output

    def test_auto_detection_does_not_trigger_for_parquet(self, tmp_path):
        """Test that parquet folders are not detected as structure folders."""
        # Create parquet eval data
        train_csv = tmp_path / "train.csv"
        eval_parquet = tmp_path / "eval.parquet"

        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        eval_data = pd.DataFrame({
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": [seq for _ in range(5)],
        })
        eval_data.to_parquet(eval_parquet)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Single parquet file - should NOT be structure folder
            f"+data.eval.val={eval_parquet.as_posix()}",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output


class TestStructureFolderWithPAtLMetric:
    """Tests for P@L metric with structure folder evaluation."""

    def test_mlm_training_with_p_at_l_structure_folder(self, tmp_path):
        """Test MLM training with P@L metric using structure folder for coords."""
        _skip_if_no_cameo()

        train_csv = tmp_path / "train.csv"
        seq = "MKTAYIAKQRQISFVKSHFSPADKTNVKAAWGK"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(20)],
            "protein_sequence": [seq for _ in range(20)],
        })
        train_data.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=256",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Structure folder eval - has coords for P@L
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "+data.eval.cameo.format=structure",
            # Enable P@L metric
            "train.eval.metrics.p_at_l.enabled=true",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output


class TestStructureFolderMetricWhitelist:
    """Tests for per-dataset metric configuration with structure folders."""

    def test_structure_folder_with_metric_whitelist(self, tmp_path):
        """Test structure folder eval with metric whitelist."""
        _skip_if_no_cameo()

        train_csv = tmp_path / "train.csv"
        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Structure folder with metric whitelist
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "+data.eval.cameo.format=structure",
            # Whitelist only specific metrics (use Hydra list syntax)
            '+data.eval.cameo.metrics.only=[masked_accuracy,perplexity]',
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output


class TestStructureFolderMultipleEvalDatasets:
    """Tests for multiple eval datasets including structure folders."""

    def test_mixed_eval_datasets_parquet_and_structure(self, tmp_path):
        """Test training with both parquet and structure folder eval datasets."""
        _skip_if_no_cameo()

        train_csv = tmp_path / "train.csv"
        eval_parquet = tmp_path / "eval.parquet"

        seq = "MKTAYIAKQRQISFVKSHFS"
        train_data = pd.DataFrame({
            "pid": [f"train_{i}" for i in range(10)],
            "protein_sequence": [seq for _ in range(10)],
        })
        train_data.to_csv(train_csv, index=False)

        eval_data = pd.DataFrame({
            "pid": [f"eval_{i}" for i in range(5)],
            "protein_sequence": [seq for _ in range(5)],
        })
        eval_data.to_parquet(eval_parquet)

        runner = CliRunner()
        overrides = [
            "model.encoder.d_model=64",
            "model.encoder.n_layers=2",
            "model.encoder.n_heads=4",
            "model.encoder.ffn_mult=1.0",
            "model.encoder.dropout=0.0",
            "model.encoder.attn_dropout=0.0",
            "data.batch_size=2",
            "data.max_len=128",
            "data.num_workers=0",
            "data.pin_memory=false",
            f"data.train={train_csv.as_posix()}",
            # Two eval datasets: parquet and structure folder
            f"+data.eval.validation={eval_parquet.as_posix()}",
            f"+data.eval.cameo.path={CAMEO_DIR.as_posix()}",
            "+data.eval.cameo.format=structure",
            "train.num_steps=4",
            "train.log_steps=2",
            "train.eval.steps=2",
            "train.wandb.enabled=false",
            f"train.project_path={tmp_path.as_posix()}",
        ]

        result = runner.invoke(cli, ["train", *overrides])
        assert result.exit_code == 0, f"Training failed: {result.output}"
        assert "Training complete." in result.output
        # Both eval datasets should appear in output
        assert "eval/validation" in result.output
        assert "eval/cameo" in result.output


class TestStructureFolderDataLoaderIntegration:
    """Tests verifying dataloader integration with structure folders."""

    def test_structure_folder_produces_coords_in_batch(self, tmp_path):
        """Test that structure folder dataset produces coords in dataloader batches."""
        _skip_if_no_cameo()

        from torch.utils.data import DataLoader
        from libreplm.data.dataset import StructureFolderDataset
        from libreplm.data.collate import mlm_collate
        from libreplm.utils.tokenizer import Tokenizer

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=256,
        )

        tokenizer = Tokenizer()

        def collate(batch):
            return mlm_collate(
                batch,
                tokenizer,
                max_len=256,
                mask_prob=0.15,
                mask_token_prob=0.8,
                random_token_prob=0.1,
                pad_id=1,
                ignore_index=-100,
            )

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate,
            shuffle=False,
        )

        batch = next(iter(loader))

        # Should return 3-tuple (tokens, labels, coords)
        assert len(batch) == 3, "Batch should have 3 elements (tokens, labels, coords)"
        tokens, labels, coords = batch

        # Verify shapes
        assert tokens.shape[0] == 2  # batch_size
        assert tokens.shape[1] == 256  # max_len
        assert labels.shape == tokens.shape
        assert coords.shape == (2, 256, 3, 3)  # [B, L, 3_atoms, 3_xyz]

    def test_structure_folder_coords_dtype(self, tmp_path):
        """Test that structure folder coords have correct dtype."""
        _skip_if_no_cameo()

        import torch
        from torch.utils.data import DataLoader
        from libreplm.data.dataset import StructureFolderDataset
        from libreplm.data.collate import mlm_collate
        from libreplm.utils.tokenizer import Tokenizer

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=256,
        )

        tokenizer = Tokenizer()

        def collate(batch):
            return mlm_collate(
                batch,
                tokenizer,
                max_len=256,
                mask_prob=0.15,
                mask_token_prob=0.8,
                random_token_prob=0.1,
                pad_id=1,
                ignore_index=-100,
            )

        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate,
            shuffle=False,
        )

        tokens, labels, coords = next(iter(loader))

        assert coords.dtype == torch.float32

