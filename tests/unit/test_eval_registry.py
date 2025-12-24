"""Unit tests for the metric registry and factory."""

import pytest
from omegaconf import OmegaConf

from procoder.eval.base import MetricBase
from procoder.eval.registry import (
    METRIC_REGISTRY,
    build_metrics,
    get_registered_metrics,
    register_metric,
)


def test_metric_registry_not_empty():
    """Test that the registry contains metrics after import."""
    # Import to trigger registration
    import procoder.eval.metrics  # noqa: F401

    assert len(METRIC_REGISTRY) > 0
    assert "perplexity" in METRIC_REGISTRY
    assert "masked_accuracy" in METRIC_REGISTRY


def test_register_metric_decorator():
    """Test the register_metric decorator."""
    # Create a unique name to avoid conflicts
    test_name = "_test_metric_decorator"

    @register_metric(test_name)
    class TestMetric(MetricBase):
        name = test_name
        objectives = {"mlm"}

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    assert test_name in METRIC_REGISTRY
    assert METRIC_REGISTRY[test_name] is TestMetric

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_register_metric_duplicate_raises():
    """Test that registering a duplicate name raises an error."""
    test_name = "_test_duplicate"

    @register_metric(test_name)
    class TestMetric1(MetricBase):
        name = test_name

        def update(self, outputs, tokens, labels, coords, cfg):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

    with pytest.raises(ValueError, match="already registered"):

        @register_metric(test_name)
        class TestMetric2(MetricBase):
            name = test_name

            def update(self, outputs, tokens, labels, coords, cfg):
                pass

            def compute(self):
                return {}

            def reset(self):
                pass

    # Cleanup
    del METRIC_REGISTRY[test_name]


def test_get_registered_metrics():
    """Test that get_registered_metrics returns a copy of the registry."""
    registry = get_registered_metrics()
    assert isinstance(registry, dict)
    assert len(registry) > 0

    # Should be a copy, not the original
    registry["_fake"] = None
    assert "_fake" not in METRIC_REGISTRY


def test_build_metrics_for_mlm():
    """Test that build_metrics builds appropriate metrics for MLM objective."""
    cfg = OmegaConf.create(
        {
            "train": {"eval": {"metrics": {}}},
            "data": {},
            "model": {},
        }
    )

    # Build for mlm objective
    mlm_metrics = build_metrics(cfg, objective="mlm")
    mlm_names = {type(m).__name__ for m in mlm_metrics}

    # MaskedAccuracyMetric should be in mlm
    assert "MaskedAccuracyMetric" in mlm_names
    # Perplexity should be available
    assert "PerplexityMetric" in mlm_names


def test_build_metrics_respects_enabled_flag():
    """Test that build_metrics respects the enabled flag."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": False},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {},
        }
    )

    metrics = build_metrics(cfg, objective="mlm")
    metric_names = {type(m).__name__ for m in metrics}

    # Masked accuracy disabled, perplexity enabled
    assert "MaskedAccuracyMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_filters_by_coords_requirement():
    """Test that metrics requiring coords are filtered when coords unavailable."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "masked_accuracy": {"enabled": True},
                    }
                }
            },
            "data": {},
            "model": {},
        }
    )

    # Without coords
    metrics_no_coords = build_metrics(
        cfg, objective="mlm", has_coords=False
    )
    names_no_coords = {type(m).__name__ for m in metrics_no_coords}

    # PrecisionAtLMetric requires coords
    assert "PrecisionAtLMetric" not in names_no_coords
    assert "MaskedAccuracyMetric" in names_no_coords


def test_build_metrics_passes_config_params():
    """Test that metric-specific config params are passed to constructor."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {
                            "enabled": True,
                            "contact_threshold": 6.0,
                            "min_seq_sep": 8,
                        },
                    }
                }
            },
            "data": {"load_coords": True},
            "model": {},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", has_coords=True)
    p_at_l = next((m for m in metrics if type(m).__name__ == "PrecisionAtLMetric"), None)

    assert p_at_l is not None
    assert p_at_l.contact_threshold == 6.0
    assert p_at_l.min_seq_sep == 8


def test_build_metrics_only_whitelist():
    """Test that 'only' list whitelists specific metrics for a dataset."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "seq_val": {
                        "path": "/path/to/seq_val.parquet",
                        "metrics": {
                            "only": ["masked_accuracy"],  # Only masked_accuracy for this dataset
                        },
                    }
                }
            },
            "model": {},
        }
    )

    # Build for seq_val dataset with 'only' whitelist
    metrics = build_metrics(cfg, objective="mlm", eval_name="seq_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Only masked_accuracy should be enabled
    assert "MaskedAccuracyMetric" in metric_names
    assert "PerplexityMetric" not in metric_names


def test_build_metrics_only_whitelist_multiple():
    """Test 'only' list with multiple metrics."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "test_val": {
                        "path": "/path/to/test.parquet",
                        "metrics": {
                            "only": ["masked_accuracy", "perplexity"],  # Both allowed
                        },
                    }
                }
            },
            "model": {},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", eval_name="test_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Both should be enabled
    assert "MaskedAccuracyMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_only_with_override():
    """Test that per-metric enabled can override 'only' list."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "hybrid_val": {
                        "path": "/path/to/hybrid.parquet",
                        "metrics": {
                            "only": ["masked_accuracy"],  # Whitelist only masked_accuracy
                            "perplexity": {"enabled": True},  # But explicitly enable perplexity
                        },
                    }
                }
            },
            "model": {},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", eval_name="hybrid_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Both should be enabled (perplexity via override)
    assert "MaskedAccuracyMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_only_disable_override():
    """Test that per-metric enabled=false can override 'only' list inclusion."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "selective_val": {
                        "path": "/path/to/selective.parquet",
                        "metrics": {
                            "only": ["masked_accuracy", "perplexity"],  # Both in whitelist
                            "masked_accuracy": {"enabled": False},  # But disable masked_accuracy
                        },
                    }
                }
            },
            "model": {},
        }
    )

    metrics = build_metrics(cfg, objective="mlm", eval_name="selective_val")
    metric_names = {type(m).__name__ for m in metrics}

    # Only perplexity should be enabled
    assert "MaskedAccuracyMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_per_dataset_has_coords_load_coords():
    """Test per-dataset load_coords overrides global has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "struct_val": {
                        "path": "/path/to/struct.parquet",
                        "load_coords": True,  # Per-dataset: has coords
                    }
                },
            },
            "model": {},
        }
    )

    # Build for struct_val with per-dataset load_coords=True
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be enabled due to per-dataset override
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_per_dataset_has_coords_explicit():
    """Test per-dataset has_coords key (alternative to load_coords)."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": True,  # Global: has coords
                "eval": {
                    "seq_val": {
                        "path": "/path/to/seq.parquet",
                        "has_coords": False,  # Per-dataset: no coords
                    }
                },
            },
            "model": {},
        }
    )

    # Build for seq_val with per-dataset has_coords=False
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=True, eval_name="seq_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be excluded
    assert "PrecisionAtLMetric" not in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_no_per_dataset_override_uses_global():
    """Test that without per-dataset override, global has_coords is used."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": True,  # Global has coords
                "eval": {
                    "default_val": {
                        "path": "/path/to/default.parquet",
                        # No load_coords or has_coords override
                    }
                },
            },
            "model": {},
        }
    )

    # Build with global has_coords=True
    metrics = build_metrics(
        cfg, objective="mlm", has_coords=True, eval_name="default_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should use global has_coords=True
    assert "PrecisionAtLMetric" in metric_names


def test_build_metrics_combined_only_and_has_coords():
    """Test combining 'only' whitelist with per-dataset has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "masked_accuracy": {"enabled": True},
                        "perplexity": {"enabled": True},
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "eval": {
                    "seq_only": {
                        "path": "/path/to/seq.parquet",
                        "load_coords": False,
                        "metrics": {
                            "only": ["masked_accuracy", "perplexity"],
                        },
                    },
                    "struct_only": {
                        "path": "/path/to/struct.parquet",
                        "load_coords": True,
                        "metrics": {
                            "only": ["masked_accuracy", "p_at_l"],
                        },
                    },
                },
            },
            "model": {},
        }
    )

    # Sequence-only dataset
    seq_metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="seq_only"
    )
    seq_names = {type(m).__name__ for m in seq_metrics}

    assert "MaskedAccuracyMetric" in seq_names  # In 'only' list
    assert "PerplexityMetric" in seq_names  # In 'only' list

    # Structure dataset
    struct_metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_only"
    )
    struct_names = {type(m).__name__ for m in struct_metrics}

    assert "PrecisionAtLMetric" in struct_names  # In 'only' list, has coords via override


def test_build_metrics_explicit_load_coords_per_dataset():
    """Test that explicit load_coords=True enables coords for a dataset."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "struct_val": {
                        "path": "/path/to/data",
                        "load_coords": True,  # Explicit enable
                    }
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords, should be enabled due to explicit load_coords
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_no_coords_without_explicit_enable(tmp_path):
    """Test that parquet folders are not auto-detected as structure folders."""
    # Create a temporary folder with parquet files (not structure files)
    parquet_folder = tmp_path / "parquet_eval"
    parquet_folder.mkdir()
    (parquet_folder / "data.parquet").write_bytes(b"parquet placeholder")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,
                "eval": {
                    "parquet_val": {
                        "path": str(parquet_folder),
                    }
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="parquet_val"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should NOT auto-detect as structure folder (no PDB/CIF files)
    assert "PrecisionAtLMetric" not in metric_names


def test_build_metrics_structure_format_enables_coords():
    """Test that format='structure' automatically enables has_coords."""
    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "cameo": {
                        "path": "/path/to/pdb_folder",
                        "format": "structure",  # Structure format always has coords
                    }
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="cameo"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords - should be enabled due to format='structure'
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_structure_folder_auto_detected(tmp_path):
    """Test that structure folder is auto-detected and has_coords is True."""
    # Create a structure folder with PDB files
    struct_folder = tmp_path / "pdbs"
    struct_folder.mkdir()
    (struct_folder / "protein1.pdb").write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    )

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                        "perplexity": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,  # Global: no coords
                "eval": {
                    "struct_eval": {
                        "path": str(struct_folder),
                        # No explicit format or load_coords
                    }
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="struct_eval"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # PrecisionAtLMetric requires coords - should be enabled via auto-detection
    assert "PrecisionAtLMetric" in metric_names
    assert "PerplexityMetric" in metric_names


def test_build_metrics_structure_folder_string_config_auto_detected(tmp_path):
    """Test that structure folder auto-detection works with string config values."""
    # Create a structure folder with CIF files
    struct_folder = tmp_path / "cifs"
    struct_folder.mkdir()
    (struct_folder / "protein1.cif").write_text("data_\n_cell.length_a 50.0\n")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,
                "eval": {
                    # String-valued config (just a path)
                    "cif_eval": str(struct_folder),
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="cif_eval"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Should auto-detect structure folder from string path
    assert "PrecisionAtLMetric" in metric_names


def test_build_metrics_mixed_folder_parquet_takes_precedence(tmp_path):
    """Test that folders with both parquet AND structure files use parquet (no coords)."""
    # Create a folder with both parquet and PDB files (edge case)
    mixed_folder = tmp_path / "mixed"
    mixed_folder.mkdir()
    (mixed_folder / "data.parquet").write_bytes(b"parquet placeholder")
    (mixed_folder / "protein1.pdb").write_text("ATOM\nEND\n")

    cfg = OmegaConf.create(
        {
            "train": {
                "eval": {
                    "metrics": {
                        "p_at_l": {"enabled": True},
                    }
                }
            },
            "data": {
                "load_coords": False,
                "eval": {
                    "mixed_eval": {
                        "path": str(mixed_folder),
                    }
                },
            },
            "model": {},
        }
    )

    metrics = build_metrics(
        cfg, objective="mlm", has_coords=False, eval_name="mixed_eval"
    )
    metric_names = {type(m).__name__ for m in metrics}

    # Parquet takes precedence - should NOT detect as structure folder
    assert "PrecisionAtLMetric" not in metric_names
