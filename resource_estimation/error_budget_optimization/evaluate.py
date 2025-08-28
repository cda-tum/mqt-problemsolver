from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from qsharp.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts

if TYPE_CHECKING:
    from numpy.typing import NDArray


def evaluate(x: NDArray[float], y: NDArray[float], total_budget: float) -> list[float]:
    """Evaluates the impact of different error budget partitions on quantum resource estimates.

    Args:
        x: A 2D array where each row contains quantum circuit logical counts (numQubits, tCount, rotationCount, etc.).
        y: A 2D array where each row contains error budgets for logical, t_states, and rotations.
        total_budget: The total error budget to be distributed among components.

    Returns:
        List of relative differences in the product of qubits and runtime compared to
        the default budget distribution.
    """
    if len(x) != len(y):
        msg = "Input arrays X and Y must have the same number of rows"
        raise ValueError(msg)

    if x.shape[1] < 7:
        msg = "X array must have at least 7 columns for the logical counts"
        raise ValueError(msg)

    logical_count_names = [
        "numQubits",
        "tCount",
        "rotationCount",
        "rotationDepth",
        "cczCount",
        "ccixCount",
        "measurementCount",
    ]

    product_diffs = []

    for i, params in enumerate(y):
        # Create logical counts dictionary
        counts_dict = {name: int(x[i, j]) for j, name in enumerate(logical_count_names)}
        logical_counts = LogicalCounts(counts_dict)

        # Normalize parameters to the total budget
        params_sum = sum(params[:3])
        params_normalized = [param / params_sum * total_budget for param in params[:3]]

        # Create custom error budget
        custom_params = EstimatorParams()
        custom_params.error_budget = ErrorBudgetPartition()
        custom_params.error_budget.logical = params_normalized[0]
        custom_params.error_budget.t_states = params_normalized[1]
        custom_params.error_budget.rotations = params_normalized[2]

        # Use default error budget for comparison
        default_params = EstimatorParams()
        default_params.error_budget = total_budget

        # Compute estimates
        result = logical_counts.estimate(custom_params)
        default_result = logical_counts.estimate(default_params)

        # Extract resource estimates
        qubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]
        default_qubits = default_result["physicalCounts"]["physicalQubits"]
        default_runtime = default_result["physicalCounts"]["runtime"]

        # Calculate improvement (negative values indicate improvement)
        product_ratio = (qubits * runtime) / (default_qubits * default_runtime)
        product_diff = product_ratio - 1.0

        # Cap positive differences at 0 (no improvement)
        product_diffs.append(min(product_diff, 0))

    return product_diffs


def plot_results(
    product_diffs: list[float], product_diffs_optimal: list[float], legend: bool = False, bin_width: int = 4
) -> None:
    """Plots histograms comparing predicted and optimal space-time differences.

    This function visualizes the distribution of space-time differences (in percent)
    for predicted and optimal product distributions. It overlays two histograms for
    comparison and customizes axis ticks, labels, and legend.

    Args:
        product_diffs: List of space-time differences for predicted distributions.
        product_diffs_optimal: List of space-time differences for best found distributions.
        legend: Whether to display the legend on the plot. Defaults to False.
        bin_width: Width of histogram bins. Defaults to 4.
    """
    # Convert to percentages
    product_diffs_pct = [100 * diff for diff in product_diffs]
    product_diffs_optimal_pct = [100 * diff for diff in product_diffs_optimal]

    # Calculate bin edges
    all_data = product_diffs_pct + product_diffs_optimal_pct
    data_min = min(0, *all_data)  # Ensure 0 is included
    data_max = max(all_data) + bin_width
    bin_edges = np.arange(data_min, data_max + bin_width, bin_width)

    # Create plot
    _fig, ax = plt.subplots(figsize=(5, 2.5))

    # Plot histograms
    ax.hist(
        product_diffs_optimal_pct,
        bins=bin_edges,
        color="steelblue",
        edgecolor="black",
        alpha=0.5,
        label="Best Distributions Determined",
    )
    ax.hist(
        product_diffs_pct, bins=bin_edges, color="orange", edgecolor="black", alpha=0.5, label="Predicted Distributions"
    )

    # Configure plot appearance
    ax.set_xlim(data_min, data_max)
    ax.set_xticks([-100, -80, -60, -40, -20, 0])
    ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel("Space-Time Difference [%]", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)

    if legend:
        ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.show()
