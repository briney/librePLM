"""Unit tests for the structure_parser module."""

from pathlib import Path

import numpy as np
import pytest

from libreplm.utils.structure_parser import parse_structure, StructureData, AA3TO1


# Path to test data
CAMEO_DIR = Path(__file__).parent.parent / "test_data" / "cameo"


class TestParseStructure:
    """Tests for the parse_structure function."""

    @pytest.fixture
    def cameo_pdb_file(self) -> Path:
        """Return path to a CAMEO PDB file."""
        pdb_file = CAMEO_DIR / "7YPD_B.pdb"
        if not pdb_file.exists():
            pytest.skip(f"CAMEO test data not found: {pdb_file}")
        return pdb_file

    def test_parse_valid_pdb_returns_structure_data(self, cameo_pdb_file):
        """Test that parsing a valid PDB file returns StructureData."""
        result = parse_structure(cameo_pdb_file)

        assert isinstance(result, StructureData)
        assert isinstance(result.pid, str)
        assert isinstance(result.protein_sequence, str)
        assert isinstance(result.coords, np.ndarray)
        assert result.chain_id is not None

    def test_parse_valid_pdb_sequence_not_empty(self, cameo_pdb_file):
        """Test that parsed sequence is not empty."""
        result = parse_structure(cameo_pdb_file)

        assert len(result.protein_sequence) > 0
        # Sequence should only contain valid amino acid characters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        for char in result.protein_sequence:
            assert char in valid_aa, f"Invalid amino acid: {char}"

    def test_parse_valid_pdb_coords_shape(self, cameo_pdb_file):
        """Test that coords have correct shape [L, 3, 3]."""
        result = parse_structure(cameo_pdb_file)

        seq_len = len(result.protein_sequence)
        assert result.coords.shape == (seq_len, 3, 3)
        assert result.coords.dtype == np.float32

    def test_parse_valid_pdb_coords_finite(self, cameo_pdb_file):
        """Test that coords contain finite values (not NaN/Inf)."""
        result = parse_structure(cameo_pdb_file)

        # All coords should be finite for a complete backbone structure
        assert np.isfinite(result.coords).all(), "Coords should be finite"

    def test_parse_valid_pdb_pid_from_filename(self, cameo_pdb_file):
        """Test that pid is derived from filename stem."""
        result = parse_structure(cameo_pdb_file)

        assert result.pid == cameo_pdb_file.stem

    def test_parse_with_chain_id_selection(self, cameo_pdb_file):
        """Test parsing with explicit chain_id selection."""
        # Parse without chain_id first to get the default chain
        result_default = parse_structure(cameo_pdb_file)

        # Parse with explicit chain_id
        result_explicit = parse_structure(cameo_pdb_file, chain_id=result_default.chain_id)

        # Should produce same result
        assert result_explicit.protein_sequence == result_default.protein_sequence
        assert result_explicit.chain_id == result_default.chain_id

    def test_parse_invalid_chain_id_raises(self, cameo_pdb_file):
        """Test that invalid chain_id raises ValueError."""
        with pytest.raises(ValueError, match="Chain .* not found"):
            parse_structure(cameo_pdb_file, chain_id="INVALID_CHAIN")

    def test_parse_file_not_found_raises(self, tmp_path):
        """Test that non-existent file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.pdb"

        with pytest.raises(FileNotFoundError, match="not found"):
            parse_structure(fake_path)

    def test_parse_empty_structure_raises(self, tmp_path):
        """Test that empty structure raises ValueError."""
        # Create a PDB file with no amino acids (just header)
        empty_pdb = tmp_path / "empty.pdb"
        empty_pdb.write_text(
            "HEADER    EMPTY STRUCTURE\n"
            "END\n"
        )

        with pytest.raises(ValueError, match="No models found|No protein chain found|No amino acid"):
            parse_structure(empty_pdb)


class TestParseStructureMissingBackbone:
    """Tests for handling missing backbone atoms."""

    @pytest.fixture
    def pdb_missing_ca(self, tmp_path) -> Path:
        """Create a PDB file with missing CA atom."""
        pdb_file = tmp_path / "missing_ca.pdb"
        # ALA residue with CA missing
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  C   ALA A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  O   ALA A   1       2.000   1.000   0.000  1.00  0.00           O\n"
            "TER\n"
            "END\n"
        )
        return pdb_file

    def test_strict_false_fills_with_nan(self, pdb_missing_ca):
        """Test that strict=False fills missing backbone atoms with NaN."""
        result = parse_structure(pdb_missing_ca, strict=False)

        assert len(result.protein_sequence) == 1
        assert result.protein_sequence == "A"

        # CA atom (index 1) should be NaN
        assert np.isnan(result.coords[0, 1, :]).all(), "CA should be NaN"

    def test_strict_true_raises(self, pdb_missing_ca):
        """Test that strict=True raises ValueError for missing backbone atoms."""
        with pytest.raises(ValueError, match="Missing backbone atom"):
            parse_structure(pdb_missing_ca, strict=True)


class TestAminoAcidMapping:
    """Tests for amino acid 3-letter to 1-letter conversion."""

    def test_standard_amino_acids_in_mapping(self):
        """Test that all standard amino acids are in the mapping."""
        standard_aa = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        for three, one in standard_aa.items():
            assert AA3TO1.get(three) == one

    def test_non_standard_amino_acid_mapping(self):
        """Test that common non-standard residues are mapped."""
        # Selenomethionine -> Methionine
        assert AA3TO1.get("MSE") == "M"
        # Phosphoserine -> Serine
        assert AA3TO1.get("SEP") == "S"
        # Unknown -> X
        assert AA3TO1.get("UNK") == "X"

    def test_non_standard_residue_in_structure(self, tmp_path):
        """Test parsing a structure with a non-standard residue."""
        pdb_file = tmp_path / "mse.pdb"
        # MSE (selenomethionine) residue with full backbone
        pdb_file.write_text(
            "ATOM      1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  MSE A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   MSE A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "ATOM      4  O   MSE A   1       2.500   2.000   1.000  1.00  0.00           O\n"
            "TER\n"
            "END\n"
        )

        result = parse_structure(pdb_file)

        # MSE should be mapped to M (methionine)
        assert result.protein_sequence == "M"


class TestParseStructureMultipleChains:
    """Tests for structures with multiple chains."""

    @pytest.fixture
    def pdb_two_chains(self, tmp_path) -> Path:
        """Create a PDB file with two chains."""
        pdb_file = tmp_path / "two_chains.pdb"
        pdb_file.write_text(
            # Chain A - ALA
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\n"
            # Chain B - GLY
            "ATOM      4  N   GLY B   1       5.000   5.000   5.000  1.00  0.00           N\n"
            "ATOM      5  CA  GLY B   1       6.458   5.000   5.000  1.00  0.00           C\n"
            "ATOM      6  C   GLY B   1       7.000   6.500   5.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )
        return pdb_file

    def test_default_selects_first_chain(self, pdb_two_chains):
        """Test that default parsing selects the first protein chain."""
        result = parse_structure(pdb_two_chains)

        assert result.chain_id == "A"
        assert result.protein_sequence == "A"

    def test_explicit_chain_selection(self, pdb_two_chains):
        """Test selecting a specific chain."""
        result = parse_structure(pdb_two_chains, chain_id="B")

        assert result.chain_id == "B"
        assert result.protein_sequence == "G"


class TestParseStructurePathTypes:
    """Tests for different path input types."""

    def test_accepts_path_object(self, tmp_path):
        """Test that Path objects are accepted."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )

        result = parse_structure(pdb_file)  # Path object
        assert result.protein_sequence == "A"

    def test_accepts_string_path(self, tmp_path):
        """Test that string paths are accepted."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )

        result = parse_structure(str(pdb_file))  # String path
        assert result.protein_sequence == "A"


class TestParseStructureCoordOrder:
    """Tests verifying N, CA, C atom ordering in coords."""

    def test_coord_order_is_n_ca_c(self, tmp_path):
        """Test that coords are ordered as [N, CA, C]."""
        pdb_file = tmp_path / "ordered.pdb"
        # Specific coords for each atom to verify order
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )

        result = parse_structure(pdb_file)

        # coords[0] = first residue, coords[0, 0] = N, coords[0, 1] = CA, coords[0, 2] = C
        np.testing.assert_array_almost_equal(result.coords[0, 0], [1.0, 2.0, 3.0])  # N
        np.testing.assert_array_almost_equal(result.coords[0, 1], [4.0, 5.0, 6.0])  # CA
        np.testing.assert_array_almost_equal(result.coords[0, 2], [7.0, 8.0, 9.0])  # C

