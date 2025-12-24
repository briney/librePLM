"""Unit tests for the StructureFolderDataset class."""

from pathlib import Path

import numpy as np
import pytest
import torch

from procoder.data.dataset import StructureFolderDataset


# Path to test data
CAMEO_DIR = Path(__file__).parent.parent / "test_data" / "cameo"


class TestStructureFolderDatasetBasics:
    """Basic tests for StructureFolderDataset functionality."""

    @pytest.fixture
    def cameo_dataset(self) -> StructureFolderDataset:
        """Return a StructureFolderDataset using CAMEO test data."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")
        return StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=512,
        )

    def test_dataset_length_matches_file_count(self, cameo_dataset):
        """Test that dataset length matches number of PDB files."""
        # Count PDB files in CAMEO directory
        pdb_count = len(list(CAMEO_DIR.glob("*.pdb")))
        assert len(cameo_dataset) == pdb_count
        assert len(cameo_dataset) == 5  # Known count for CAMEO test data

    def test_getitem_returns_dict(self, cameo_dataset):
        """Test that __getitem__ returns a dictionary."""
        item = cameo_dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, cameo_dataset):
        """Test that returned dict has all required keys."""
        item = cameo_dataset[0]

        required_keys = {"pid", "seq", "coords", "masks", "nan_masks"}
        assert set(item.keys()) == required_keys

    def test_getitem_no_indices_key(self, cameo_dataset):
        """Test that 'indices' key is NOT present (structure files don't have VQ indices)."""
        item = cameo_dataset[0]
        assert "indices" not in item

    def test_pid_is_string(self, cameo_dataset):
        """Test that pid is a string."""
        item = cameo_dataset[0]
        assert isinstance(item["pid"], str)
        assert len(item["pid"]) > 0

    def test_seq_is_string(self, cameo_dataset):
        """Test that seq is a string of amino acids."""
        item = cameo_dataset[0]
        assert isinstance(item["seq"], str)
        assert len(item["seq"]) > 0

        # Verify valid amino acid characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        for char in item["seq"]:
            assert char in valid_aa

    def test_coords_shape(self, cameo_dataset):
        """Test that coords have shape [max_length, 3, 3]."""
        item = cameo_dataset[0]
        coords = item["coords"]

        assert isinstance(coords, torch.Tensor)
        assert coords.shape == (512, 3, 3)  # max_length, 3 atoms, 3 xyz
        assert coords.dtype == torch.float32

    def test_masks_shape_and_dtype(self, cameo_dataset):
        """Test that masks have correct shape and dtype."""
        item = cameo_dataset[0]

        masks = item["masks"]
        assert isinstance(masks, torch.Tensor)
        assert masks.shape == (512,)  # max_length
        assert masks.dtype == torch.bool

    def test_nan_masks_equals_masks(self, cameo_dataset):
        """Test that nan_masks equals masks (for complete backbones)."""
        item = cameo_dataset[0]

        # For structures with complete backbones, nan_masks == masks
        assert torch.equal(item["masks"], item["nan_masks"])

    def test_has_coords_attribute_is_true(self, cameo_dataset):
        """Test that has_coords attribute is True."""
        assert cameo_dataset.has_coords is True

    def test_repr_contains_useful_info(self, cameo_dataset):
        """Test that __repr__ contains useful information."""
        repr_str = repr(cameo_dataset)

        assert "StructureFolderDataset" in repr_str
        assert str(CAMEO_DIR) in repr_str or "cameo" in repr_str
        assert "num_files" in repr_str
        assert "max_length" in repr_str


class TestStructureFolderDatasetPadding:
    """Tests for padding behavior."""

    @pytest.fixture
    def short_max_length_dataset(self) -> StructureFolderDataset:
        """Dataset with very short max_length for truncation testing."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")
        return StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=50,  # Very short to force truncation
        )

    def test_coords_padded_with_nan(self):
        """Test that coords are padded with NaN beyond sequence length."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=1024,  # Long enough to see padding
        )
        item = dataset[0]

        seq_len = len(item["seq"])
        coords = item["coords"]

        # Valid positions should not have NaN
        valid_coords = coords[:seq_len]
        assert torch.isfinite(valid_coords).all(), "Valid positions should be finite"

        # Padded positions should be NaN
        if seq_len < 1024:
            padded_coords = coords[seq_len:]
            assert torch.isnan(padded_coords).all(), "Padded positions should be NaN"

    def test_masks_false_for_padded_positions(self):
        """Test that masks are False for padded positions."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=1024,
        )
        item = dataset[0]

        seq_len = len(item["seq"])
        masks = item["masks"]

        # Valid positions should be True
        assert masks[:seq_len].all()

        # Padded positions should be False
        if seq_len < 1024:
            assert not masks[seq_len:].any()

    def test_truncation_for_long_sequences(self, short_max_length_dataset):
        """Test that sequences are truncated to max_length."""
        item = short_max_length_dataset[0]

        # Sequence should be truncated
        assert len(item["seq"]) <= 50

        # Coords should have max_length rows
        assert item["coords"].shape[0] == 50

        # All masks should be True (no padding after truncation)
        assert item["masks"].all()


class TestStructureFolderDatasetRecursive:
    """Tests for recursive directory searching."""

    @pytest.fixture
    def nested_pdb_dir(self, tmp_path) -> Path:
        """Create a directory with nested subdirectories containing PDB files."""
        # Create root dir
        root = tmp_path / "structures"
        root.mkdir()

        # Create PDB in root
        (root / "root.pdb").write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\nEND\n"
        )

        # Create subdir with PDB
        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "nested.pdb").write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   GLY A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\nEND\n"
        )

        return root

    def test_recursive_false_excludes_subdirs(self, nested_pdb_dir):
        """Test that recursive=False only finds files in root directory."""
        dataset = StructureFolderDataset(
            folder_path=str(nested_pdb_dir),
            max_length=64,
            recursive=False,
        )

        assert len(dataset) == 1

    def test_recursive_true_includes_subdirs(self, nested_pdb_dir):
        """Test that recursive=True finds files in subdirectories."""
        dataset = StructureFolderDataset(
            folder_path=str(nested_pdb_dir),
            max_length=64,
            recursive=True,
        )

        assert len(dataset) == 2


class TestStructureFolderDatasetChainId:
    """Tests for chain_id parameter."""

    @pytest.fixture
    def multi_chain_pdb(self, tmp_path) -> Path:
        """Create a PDB file with multiple chains."""
        pdb_file = tmp_path / "multi_chain.pdb"
        pdb_file.write_text(
            # Chain A - ALA
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\n"
            # Chain B - GLY GLY
            "ATOM      4  N   GLY B   1       5.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      5  CA  GLY B   1       6.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      6  C   GLY B   1       7.000   1.500   0.000  1.00  0.00           C\n"
            "ATOM      7  N   GLY B   2       8.000   1.500   0.000  1.00  0.00           N\n"
            "ATOM      8  CA  GLY B   2       9.458   1.500   0.000  1.00  0.00           C\n"
            "ATOM      9  C   GLY B   2      10.000   3.000   0.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )
        return tmp_path

    def test_chain_id_selects_specific_chain(self, multi_chain_pdb):
        """Test that chain_id parameter selects the specified chain."""
        # Select chain B (has 2 GLY residues)
        dataset = StructureFolderDataset(
            folder_path=str(multi_chain_pdb),
            max_length=64,
            chain_id="B",
        )

        item = dataset[0]
        assert item["seq"] == "GG"


class TestStructureFolderDatasetErrors:
    """Tests for error handling."""

    def test_empty_folder_raises_value_error(self, tmp_path):
        """Test that empty folder raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No structure files found"):
            StructureFolderDataset(
                folder_path=str(empty_dir),
                max_length=64,
            )

    def test_not_a_directory_raises_value_error(self, tmp_path):
        """Test that file path (not directory) raises ValueError."""
        # Create a file instead of directory
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")

        with pytest.raises(ValueError, match="Not a directory"):
            StructureFolderDataset(
                folder_path=str(file_path),
                max_length=64,
            )

    def test_folder_with_non_structure_files_only(self, tmp_path):
        """Test that folder with no structure files raises ValueError."""
        no_pdb_dir = tmp_path / "no_pdbs"
        no_pdb_dir.mkdir()

        # Create non-PDB files
        (no_pdb_dir / "data.csv").write_text("a,b,c\n1,2,3")
        (no_pdb_dir / "readme.txt").write_text("readme")

        with pytest.raises(ValueError, match="No structure files found"):
            StructureFolderDataset(
                folder_path=str(no_pdb_dir),
                max_length=64,
            )


class TestStructureFolderDatasetFileExtensions:
    """Tests for different file extension support."""

    @pytest.fixture
    def mixed_extensions_dir(self, tmp_path) -> Path:
        """Create a directory with various structure file extensions."""
        struct_dir = tmp_path / "structures"
        struct_dir.mkdir()

        pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\nEND\n"
        )

        # Create files with different extensions
        (struct_dir / "file1.pdb").write_text(pdb_content)
        (struct_dir / "file2.ent").write_text(pdb_content)

        return struct_dir

    def test_supports_pdb_extension(self, mixed_extensions_dir):
        """Test that .pdb files are discovered."""
        dataset = StructureFolderDataset(
            folder_path=str(mixed_extensions_dir),
            max_length=64,
        )

        # Should find both .pdb and .ent files
        assert len(dataset) == 2

    def test_supports_ent_extension(self, mixed_extensions_dir):
        """Test that .ent files are discovered."""
        # .ent is a common alternative extension for PDB files
        dataset = StructureFolderDataset(
            folder_path=str(mixed_extensions_dir),
            max_length=64,
        )

        pids = [dataset[i]["pid"] for i in range(len(dataset))]
        assert "file2" in pids


class TestStructureFolderDatasetIntegration:
    """Integration tests with real CAMEO data."""

    def test_all_cameo_files_load_successfully(self):
        """Test that all CAMEO test files load without errors."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=512,
        )

        # Load all items - should not raise
        for i in range(len(dataset)):
            item = dataset[i]
            assert "pid" in item
            assert "seq" in item
            assert "coords" in item
            assert len(item["seq"]) > 0

    def test_cameo_pids_match_filenames(self):
        """Test that PIDs match the PDB filenames."""
        if not CAMEO_DIR.exists():
            pytest.skip(f"CAMEO test data not found: {CAMEO_DIR}")

        dataset = StructureFolderDataset(
            folder_path=str(CAMEO_DIR),
            max_length=512,
        )

        expected_pids = {p.stem for p in CAMEO_DIR.glob("*.pdb")}
        actual_pids = {dataset[i]["pid"] for i in range(len(dataset))}

        assert actual_pids == expected_pids

