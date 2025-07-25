from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qsharp.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts


def evaluate(X, Y, total_budget):
    """
    Evaluates the impact of different error budget partitions on quantum resource estimates.
    Args:
        X: A 2D array where each row contains quantum circuit logical counts (e.g., numQubits, tCount, rotationCount, etc.).
        Y: A 2D array where each row contains error budgetds for logical, t_states, and rotations.
        total_budget: The total error budget to be distributed among logical, t_states, and rotations.
    Returns:
        qubits_diffs: List of relative differences in physical qubits compared to the default budget distribution.
        runtime_diffs: List of relative differences in runtime compared to the default budget distribution.
        product_diffs: List of relative differences in the product of qubits and runtime compared to the default budget distribution.
        qubits_list: List of estimated physical qubits for each parameter set.
        runtime_list: List of estimated runtimes for each parameter set.
        default_qubits_list: List of physical qubits using the default budget for each parameter set.
        default_runtime_list: List of runtimes using the default budget for each parameter set.
    """

    product_diffs = []
    for i, params in enumerate(Y):
        c = {}
        c["numQubits"] = int(X[i, 0])
        c["tCount"] = int(X[i, 1])
        c["rotationCount"] = int(X[i, 2])
        c["rotationDepth"] = int(X[i, 3])
        c["cczCount"] = int(X[i, 4])
        c["ccixCount"] = int(X[i, 5])
        c["measurementCount"] = int(X[i, 6])
        logical_counts = LogicalCounts(c)
        params_sum = params[0] + params[1] + params[2]
        params_normalized = [
            params[0] / params_sum * total_budget,
            params[1] / params_sum * total_budget,
            params[2] / params_sum * total_budget,
        ]

        parameters = EstimatorParams()
        parameters.error_budget = ErrorBudgetPartition()
        parameters.error_budget.logical = params_normalized[0]
        parameters.error_budget.t_states = params_normalized[1]
        parameters.error_budget.rotations = params_normalized[2]

        default_parameters = EstimatorParams()
        default_parameters.error_budget = total_budget

        result = logical_counts.estimate(parameters)
        default_result = logical_counts.estimate(default_parameters)
        qubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]
        default_qubits = default_result["physicalCounts"]["physicalQubits"]
        default_runtime = default_result["physicalCounts"]["runtime"]

        product_diff = ((qubits * runtime) - (default_qubits * default_runtime)) / (default_qubits * default_runtime)
        if product_diff > 0:
            product_diff = 0

        product_diffs.append(product_diff)

    return product_diffs


def plot_results(product_diffs, product_diffs_optimal, legend=False, bin_width=4):
    """
    Plots histograms comparing predicted and optimal space-time differences.
    This function visualizes the distribution of space-time differences (in percent)
    for predicted and optimal product distributions. It overlays two histograms for
    comparison and customizes axis ticks, labels, and legend.
    Args:
        product_diffs: List of space-time differences for predicted distributions.
        product_diffs_optimal: List of space-time differences for best found distributions.
        legend: Whether to display the legend on the plot. Defaults to False.
        bin_width: Width of histogram bins. Defaults to 4.
    Returns:
        None. Displays the plot.
    """

    product_diffs = [100 * i for i in product_diffs]
    product_diffs_optimal = [100 * i for i in product_diffs_optimal]

    all_data = product_diffs + product_diffs_optimal
    data_min = min(all_data)
    data_max = max(all_data)

    data_min = min(0, data_min)
    data_max += bin_width

    bin_edges = np.arange(data_min, data_max + bin_width, bin_width)

    np.arange(-100, 1, 20)
    _fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(
        product_diffs_optimal,
        bins=bin_edges,
        color="steelblue",
        edgecolor="black",
        alpha=0.5,
        label="Best Distributions Determined",
    )
    ax.hist(
        product_diffs, bins=bin_edges, color="orange", edgecolor="black", alpha=0.5, label="Predicted Distributions"
    )
    ax.set_xlim(data_min, data_max)
    ax.set_xticks([-100, -80, -60, -40, -20, 0])
    ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel("Space-Time Difference [%]", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    if legend:
        ax.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()
