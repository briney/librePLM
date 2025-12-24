"""Evaluator class for orchestrating metric computation."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from libreplm.eval.base import Metric
from libreplm.eval.registry import build_metrics


def _get_model_device(model: nn.Module, accelerator) -> torch.device:
    """Resolve the device to place tensors on."""
    if accelerator is not None:
        return accelerator.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _unwrap_model(model: nn.Module, accelerator) -> nn.Module:
    """Unwrap model from Accelerate/DDP if needed."""
    return accelerator.unwrap_model(model) if accelerator is not None else model


class Evaluator:
    """Orchestrates evaluation metric computation.

    The Evaluator handles:
    - Building metrics based on configuration and available resources
    - Running evaluation forward passes
    - Distributed aggregation of metric state
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        accelerator,
    ):
        """Initialize the evaluator.

        Args:
            cfg: Full configuration object.
            model: The model to evaluate.
            accelerator: Accelerate accelerator instance (or None).
        """
        self.cfg = cfg
        self.model = model
        self.accelerator = accelerator

        # Determine objective and resource availability
        self.objective = "mlm"  # MLM-only after refactoring
        self.has_coords = bool(cfg.data.get("load_coords", False))

        # Cache for metrics per eval dataset
        self._metrics_cache: dict[str, list[Metric]] = {}

        # Cache for whether attention weights are needed per eval dataset
        self._needs_attentions_cache: dict[str, bool] = {}

    def _get_metrics(self, eval_name: str | None = None) -> list[Metric]:
        """Get or build metrics for an eval dataset.

        Args:
            eval_name: Name of the eval dataset (for per-dataset overrides).

        Returns:
            List of metric instances.
        """
        cache_key = eval_name or "__default__"
        if cache_key not in self._metrics_cache:
            metrics = build_metrics(
                cfg=self.cfg,
                objective=self.objective,
                has_coords=self.has_coords,
                eval_name=eval_name,
            )
            self._metrics_cache[cache_key] = metrics

            # Check if any metric needs attention weights (e.g., p_at_l)
            self._needs_attentions_cache[cache_key] = any(
                getattr(m, "name", "") == "p_at_l" for m in metrics
            )

        return self._metrics_cache[cache_key]

    def _needs_attentions(self, eval_name: str | None = None) -> bool:
        """Check if attention weights are needed for an eval dataset.

        Args:
            eval_name: Name of the eval dataset.

        Returns:
            True if any metric for this dataset needs attention weights.
        """
        cache_key = eval_name or "__default__"
        # Ensure metrics are built (which populates the cache)
        if cache_key not in self._needs_attentions_cache:
            self._get_metrics(eval_name)
        return self._needs_attentions_cache.get(cache_key, False)

    def _gather_metric_states(self, metrics: list[Metric]) -> None:
        """Aggregate metric states across distributed processes.

        Supports two gathering modes:
        1. Tensor-based gathering (default): For metrics with fixed-size state.
           Uses accelerator.gather_for_metrics() which requires identical shapes.
        2. Object-based gathering: For metrics with variable-length state.
           Uses accelerator.gather_object() which handles arbitrary Python objects.

        Args:
            metrics: List of metrics to aggregate.
        """
        if self.accelerator is None:
            return

        device = self.accelerator.device

        for metric in metrics:
            # Check if metric uses object-based gathering (for variable-length data)
            if hasattr(metric, "state_objects"):
                objects = metric.state_objects()
                if objects is not None:
                    # Use gather_object for variable-length data like lists of dicts
                    # gather_object is a standalone function from accelerate.utils
                    gathered = gather_object(objects)
                    # gathered is a list containing state_objects() from each process
                    if hasattr(metric, "load_state_objects"):
                        metric.load_state_objects(gathered)
                    continue

            # Fall back to tensor-based gathering for fixed-size state
            state_tensors = metric.state_tensors()
            if not state_tensors:
                continue

            # Gather and sum each state tensor separately
            gathered_tensors = []
            for t in state_tensors:
                t_device = t.to(device)
                gathered = self.accelerator.gather_for_metrics(t_device)

                # Sum across processes
                # Accelerate's gather_for_metrics concatenates tensors along dim=0,
                # so a tensor of shape [2] becomes [N*2] with N processes (flattened).
                # We need to reshape back to [N, *original_shape] before summing.
                original_size = t_device.numel()
                gathered_size = gathered.numel()

                if gathered_size == original_size:
                    # Single process, no aggregation needed
                    summed = gathered
                else:
                    # Multi-process: reshape to [num_processes, *original_shape] then sum
                    num_processes = gathered_size // original_size
                    reshaped = gathered.view(num_processes, *t_device.shape)
                    summed = reshaped.sum(dim=0)

                gathered_tensors.append(summed)

            metric.load_state_tensors(gathered_tensors)

    def evaluate(
        self,
        eval_loader: DataLoader,
        eval_name: str,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Args:
            eval_loader: DataLoader for the evaluation dataset.
            eval_name: Name of the evaluation dataset.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = self._get_metrics(eval_name)

        # Reset all metrics
        for metric in metrics:
            metric.reset()

        # Check if any metrics need attention weights (e.g., p_at_l)
        needs_attentions = self._needs_attentions(eval_name)

        self.model.eval()
        ignore_index = int(self.cfg.model.classifier.get("ignore_index", -100))

        with torch.no_grad():
            for batch in eval_loader:
                # Unpack batch
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    tokens, labels, coords = batch
                else:
                    tokens, labels = batch
                    coords = None

                # Move to device if not using accelerator
                if self.accelerator is None:
                    device = _get_model_device(self.model, self.accelerator)
                    tokens = tokens.to(device)
                    labels = labels.to(device)
                    if coords is not None:
                        coords = coords.to(device)

                # Forward pass (request attention weights if needed for metrics like p_at_l)
                outputs = self.model(
                    tokens=tokens,
                    labels=labels,
                    ignore_index=ignore_index,
                    output_attentions=needs_attentions,
                )

                # Update all metrics
                for metric in metrics:
                    try:
                        metric.update(outputs, tokens, labels, coords, self.cfg)
                    except Exception as e:
                        warnings.warn(f"Metric '{metric.name}' update failed: {e}")

        # Aggregate across distributed processes
        self._gather_metric_states(metrics)

        # Compute final metric values
        results: dict[str, float] = {}
        for metric in metrics:
            try:
                computed = metric.compute()
                results.update(computed)
            except Exception as e:
                warnings.warn(f"Metric '{metric.name}' compute failed: {e}")

        self.model.train()
        return results

    def evaluate_all(
        self,
        eval_loaders: dict[str, DataLoader],
    ) -> dict[str, dict[str, float]]:
        """Run evaluation on all datasets.

        Args:
            eval_loaders: Dictionary mapping dataset names to DataLoaders.

        Returns:
            Dictionary mapping dataset names to their metric results.
        """
        all_results: dict[str, dict[str, float]] = {}
        for eval_name, eval_loader in eval_loaders.items():
            all_results[eval_name] = self.evaluate(eval_loader, eval_name)
        return all_results

    def clear_cache(self) -> None:
        """Clear the metrics cache (e.g., after config changes)."""
        self._metrics_cache.clear()
        self._needs_attentions_cache.clear()
