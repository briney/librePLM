"""Modular evaluation metrics system for librePLM training."""

from procoder.eval.base import Metric, MetricBase
from procoder.eval.evaluator import Evaluator
from procoder.eval.logger import MetricLogger
from procoder.eval.registry import METRIC_REGISTRY, build_metrics, register_metric

# Import metrics to ensure they're registered
import procoder.eval.metrics  # noqa: F401

__all__ = [
    "Metric",
    "MetricBase",
    "Evaluator",
    "MetricLogger",
    "METRIC_REGISTRY",
    "build_metrics",
    "register_metric",
]

