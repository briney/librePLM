"""Modular evaluation metrics system for librePLM training."""

from libreplm.eval.base import Metric, MetricBase
from libreplm.eval.evaluator import Evaluator
from libreplm.eval.logger import MetricLogger
from libreplm.eval.registry import METRIC_REGISTRY, build_metrics, register_metric

# Import metrics to ensure they're registered
import libreplm.eval.metrics  # noqa: F401

__all__ = [
    "Metric",
    "MetricBase",
    "Evaluator",
    "MetricLogger",
    "METRIC_REGISTRY",
    "build_metrics",
    "register_metric",
]

