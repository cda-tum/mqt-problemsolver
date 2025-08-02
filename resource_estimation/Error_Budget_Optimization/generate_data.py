from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qsharp.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts
from qsharp.interop.qiskit import estimate
from tqdm import tqdm


def find_optimized_budgets(total_budget, num_iterations, counts):
    """
    Randomly distributes the total error budget among logical, T-state, and rotation errors,
    and estimates the physical resource requirements for each distribution. Tracks the distribution
    that yields the lowest product of runtime and physical qubits.

    Args:
        total_budget: The total error budget to be distributed.
        num_iterations: Number of random distributions to try.
        counts: LogicalCounts object containing circuit logical counts.

    Returns:
        A tuple containing:
            - List of optimal logical, T-state, and rotation budgets found.
            - The best metric found (runtime * physical qubits).
            - The default metric (runtime * physical qubits with default partition).
    """
    parameters = EstimatorParams()
    default_parameters = EstimatorParams()
    default_parameters.error_budget = total_budget

    default_result = counts.estimate(default_parameters)
    default_result["logicalCounts"]

    default_physicalqubits = default_result["physicalCounts"]["physicalQubits"]
    default_runtime = default_result["physicalCounts"]["runtime"]

    running_metric = None

    running_optimal_parameters = {}
    logical_cache = []
    t_cache = []
    rotation_cache = []

    for _i in range(num_iterations):
        parameters.error_budget = ErrorBudgetPartition()
        logical_budget_random = np.random.uniform(0, 1)
        t_budget_random = np.random.uniform(0, 1)
        rotation_budget_random = np.random.uniform(0, 1)
        budget_sum = logical_budget_random + t_budget_random + rotation_budget_random

        logical_budget = (logical_budget_random / budget_sum) * total_budget
        t_budget = (t_budget_random / budget_sum) * total_budget
        rotation_budget = (rotation_budget_random / budget_sum) * total_budget

        parameters.error_budget.logical = logical_budget
        parameters.error_budget.t_states = t_budget
        parameters.error_budget.rotations = rotation_budget

        result = counts.estimate(params=parameters)
        default_result = counts.estimate()

        physicalqubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]

        default_metric = default_runtime * default_physicalqubits
        current_metric = runtime * physicalqubits

        if running_metric is None:
            running_metric = current_metric
            logical_cache = logical_budget
            t_cache = t_budget
            rotation_cache = rotation_budget

        if current_metric < running_metric:
            running_metric = current_metric
            logical_cache = logical_budget
            t_cache = t_budget
            rotation_cache = rotation_budget

    running_optimal_parameters["logical_budget"] = logical_cache
    running_optimal_parameters["t_budget"] = t_cache
    running_optimal_parameters["rotation_budget"] = rotation_cache

    return list(running_optimal_parameters.values()), running_metric, default_metric


def generate_data(total_error_budget, counts, path="MQTBench"):
    """
    Generates a dataset consisting of logical counts of quantum circuits and respective optimized error budgets.

    This function searches for QASM files in the "MQTBench" directory, loads each circuit,
    estimates its logical counts, and computes optimized error budget partitions using random sampling.
    For each circuit, it collects relevant counts and the optimal error budget distribution,
    appending the results to a list.

    Args:
        total_error_budget: The total error budget to be distributed among error types.
        counts: LogicalCounts object or dictionary containing logical counts for estimation.

    Returns:
        A list of lists, where each inner list contains circuit-specific counts and the corresponding
        optimized error budget partition.
    """
    qasm_files = [Path(root) / file for root, _, files in os.walk(path) for file in files if file.endswith(".qasm")]
    results = []

    for file in tqdm(qasm_files):
        with Path.open(file, encoding="utf-8") as f:
            qasm = f.read()
            qc = QuantumCircuit.from_qasm_str(qasm)
        try:
            estimation = estimate(qc)
            counts = estimation["logicalCounts"]
            if counts["rotationCount"] == 0:
                continue
            counts = LogicalCounts(counts)
            combinations, _running_metric, _default_metric = find_optimized_budgets(total_error_budget, 1000, counts)
            specific_data = [
                int(counts["numQubits"]),
                int(counts["tCount"]),
                int(counts["rotationCount"]),
                int(counts["rotationDepth"]),
                int(counts["cczCount"]),
                int(counts["ccixCount"]),
                int(counts["measurementCount"]),
            ]
            specific_data += combinations
            results.append(specific_data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
        clear_output(wait=True)
    return results
