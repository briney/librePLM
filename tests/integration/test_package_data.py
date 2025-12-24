"""Tests for packaged data files."""

import importlib.resources as r


def test_packaged_configs_exist():
    """Test that packaged config files exist."""
    root = r.files("procoder")
    assert (root / "configs" / "config.yaml").is_file()
