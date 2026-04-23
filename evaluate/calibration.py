"""Calibration analysis: Expected Calibration Error and reliability diagrams.

ECE measures whether model confidence scores match empirical accuracy.
A perfectly calibrated model has ECE = 0.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from shared.logging_utils import JSONLogger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Compute Expected Calibration Error and generate reliability diagrams.

    Args:
        n_bins:      Number of equal-width confidence bins over [0, 1].
        json_logger: Optional JSONLogger for structured warning logging.
    """

    def __init__(
        self,
        n_bins: int = 10,
        json_logger: Optional[JSONLogger] = None,
    ) -> None:
        self.n_bins = n_bins
        self.json_logger = json_logger

    def _warn(self, message: str) -> None:
        if self.json_logger is not None:
            self.json_logger.log({
                "type": "warning",
                "component": "CalibrationAnalyzer",
                "message": message,
            })
        else:
            _logger.warning(message)

    def compute_ece(
        self,
        confidences: list[float],
        labels: list[int],
    ) -> float | None:
        """Compute Expected Calibration Error.

        Partitions (confidence, label) pairs into ``n_bins`` equal-width bins
        over [0, 1] and returns the weighted mean absolute difference between
        mean confidence and empirical accuracy per bin.

        Args:
            confidences: Model confidence scores, each in [0, 1].
            labels:      Ground-truth binary labels (0 or 1).

        Returns:
            ECE in [0, 1], or None if ``confidences`` is empty.
        """
        if not confidences:
            self._warn("Empty confidences list; cannot compute ECE.")
            return None

        n = len(confidences)
        bin_width = 1.0 / self.n_bins

        ece = 0.0
        for b in range(self.n_bins):
            lo = b * bin_width
            hi = lo + bin_width
            # Include upper boundary in the last bin
            if b == self.n_bins - 1:
                indices = [i for i, c in enumerate(confidences) if lo <= c <= hi]
            else:
                indices = [i for i, c in enumerate(confidences) if lo <= c < hi]

            if not indices:
                continue

            bin_count = len(indices)
            bin_weight = bin_count / n
            mean_conf = sum(confidences[i] for i in indices) / bin_count
            accuracy = sum(labels[i] for i in indices) / bin_count
            ece += bin_weight * abs(mean_conf - accuracy)

        return ece

    def plot_reliability_diagram(
        self,
        confidences: list[float],
        labels: list[int],
        model_name: str,
        save_dir: str,
    ) -> str | None:
        """Generate and save a reliability diagram PNG.

        Plots mean confidence vs. empirical accuracy per bin, with a diagonal
        reference line representing perfect calibration.

        Args:
            confidences: Model confidence scores, each in [0, 1].
            labels:      Ground-truth binary labels (0 or 1).
            model_name:  Used in the plot title and filename.
            save_dir:    Directory where the PNG is saved.

        Returns:
            Absolute path to the saved PNG, or None if matplotlib is unavailable
            or confidences is empty.
        """
        if not confidences:
            self._warn(f"Empty confidences for {model_name}; skipping reliability diagram.")
            return None

        if not _HAS_MPL:
            self._warn("matplotlib not available; skipping reliability diagram.")
            return None

        n = len(confidences)
        bin_width = 1.0 / self.n_bins
        bin_centers: list[float] = []
        mean_confs: list[float] = []
        accuracies: list[float] = []
        bin_counts: list[int] = []

        for b in range(self.n_bins):
            lo = b * bin_width
            hi = lo + bin_width
            if b == self.n_bins - 1:
                indices = [i for i, c in enumerate(confidences) if lo <= c <= hi]
            else:
                indices = [i for i, c in enumerate(confidences) if lo <= c < hi]

            if not indices:
                continue

            bin_centers.append(lo + bin_width / 2)
            mean_confs.append(sum(confidences[i] for i in indices) / len(indices))
            accuracies.append(sum(labels[i] for i in indices) / len(indices))
            bin_counts.append(len(indices))

        fig, ax = plt.subplots(figsize=(6, 6))

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        # Reliability curve
        if mean_confs:
            ax.bar(
                bin_centers,
                accuracies,
                width=bin_width * 0.8,
                alpha=0.6,
                color="steelblue",
                label="Empirical accuracy",
            )
            ax.plot(mean_confs, accuracies, "ro-", markersize=5, label="Mean confidence")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        safe_name = model_name.replace("/", "_").replace(".", "_")
        ax.set_title(f"Reliability Diagram — {model_name}")
        ax.legend(fontsize=8)
        fig.tight_layout()

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(save_dir) / f"reliability_{safe_name}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        return out_path
