from __future__ import annotations

import copy
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit
from qsharp.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts
from qsharp.interop.qiskit import estimate


def generate_benchmarks(benchmarks_and_sizes: list[tuple[str, list[int]]]) -> list[QuantumCircuit]:
    """
    Generates a list of benchmarks with their respective sizes.

    Args:
        benchmarks_and_sizes: A list containing tuples each containing the benchmark name and a list of sizes.

    Returns:
        A list of tuples, each containing the benchmark name and its corresponding sizes.
    """
    circuits = []
    for benchmark, sizes in benchmarks_and_sizes:
        for size in sizes:
            circuit = get_benchmark(benchmark=benchmark, circuit_size=size, level=BenchmarkLevel.INDEP)
            circuits.append(circuit)
    return circuits


def find_optimized_budgets(
    total_budget: float, num_iterations: int, counts: LogicalCounts
) -> tuple[list[float], int, int]:
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

    default_parameters = EstimatorParams()
    default_parameters.error_budget = total_budget

    default_result = counts.estimate(default_parameters)

    default_physicalqubits = default_result["physicalCounts"]["physicalQubits"]
    default_runtime = default_result["physicalCounts"]["runtime"]
    default_metric = default_runtime * default_physicalqubits

    best_metric = math.inf

    for _i in range(num_iterations):
        parameters = EstimatorParams()
        parameters.error_budget = ErrorBudgetPartition()

        rng = np.random.default_rng()
        low = np.nextafter(0, 1)  # Exclude zero to avoid zero error budgets
        budgets = rng.uniform(low, 1, size=3)
        budgets = budgets / np.sum(budgets)  # Normalize to sum to 1

        parameters.error_budget.logical = float(budgets[0] * total_budget)
        parameters.error_budget.t_states = float(budgets[1] * total_budget)
        parameters.error_budget.rotations = float(budgets[2] * total_budget)

        result = counts.estimate(params=parameters)

        physicalqubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]

        current_metric = runtime * physicalqubits

        if current_metric < best_metric:
            best_metric = current_metric
            best_parameters = copy.deepcopy(parameters)

    return (
        [
            best_parameters.error_budget.logical,
            best_parameters.error_budget.t_states,
            best_parameters.error_budget.rotations,
        ],
        int(best_metric),
        int(default_metric),
    )


def generate_data(
    total_error_budget: float,
    number_of_randomly_generated_distributions: int,
    benchmarks: list[QuantumCircuit] | None = None,
    path: str | None = None,
) -> list[OrderedDict[str, float | int]]:
    """
    Generates a dataset consisting of logical counts of quantum circuits and respective optimized error budgets.

    This function searches for QASM files in the given directory, loads each circuit,
    estimates its logical counts, and computes optimized error budget partitions using random sampling.
    For each circuit, it collects relevant counts and the optimal error budget distribution,
    appending the results to a list.

    Args:
        total_error_budget: The total error budget to be distributed among error types.

    Returns:
        A list of lists, where each inner list contains circuit-specific counts and the corresponding
        optimized error budget partition.
    """

    if path:
        qasm_files = [Path(root) / file for root, _, files in os.walk(path) for file in files if file.endswith(".qasm")]
        circuits = [QuantumCircuit.from_qasm_file(file) for file in qasm_files]
    elif benchmarks:
        circuits = benchmarks
    else:
        msg = "Either 'path' or 'benchmarks' must be provided."
        raise ValueError(msg)
    results = []

    for qc in circuits:
        try:
            estimation = estimate(qc)
            counts = estimation["logicalCounts"]
            if counts["rotationCount"] == 0:
                continue  # Skip circuits without rotations, as we want to ensure distributing error budgets among all three types.
            counts = LogicalCounts(counts)
            combinations, _running_metric, _default_metric = find_optimized_budgets(
                total_error_budget, number_of_randomly_generated_distributions, counts
            )
            specific_data = OrderedDict()
            for key, value in counts.items():
                specific_data[key] = value
            specific_data["logical"] = combinations[0]
            specific_data["t_states"] = combinations[1]
            specific_data["rotations"] = combinations[2]
            results.append(specific_data)
        except Exception as e:
            print(f"Error processing the current circuit: {qc.name}. Error: {e}")
            continue
        # clear_output(wait=True)
    return results
