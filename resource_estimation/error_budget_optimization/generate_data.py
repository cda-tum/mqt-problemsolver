from __future__ import annotations

import copy
import math
import os
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, qasm2, transpile
from qsharp.estimator import ErrorBudgetPartition, EstimatorParams, LogicalCounts
from qsharp.interop.qiskit import estimate

if TYPE_CHECKING:
    from collections.abc import Iterator

QISKIT_STD_GATES = [
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "cp",
    "crx",
    "cry",
    "crz",
    "ch",
    "ccx",
    "cu",
    "id",
    "u1",
    "u2",
    "u3",
]  # Removing swap gates and non-standard gates to ensure compatibility with the estimator.


def generate_benchmarks(benchmarks_and_sizes: list[tuple[str, list[int]]]) -> Iterator[QuantumCircuit]:
    """Generates a list of benchmarks with their respective sizes.

    Args:
        benchmarks_and_sizes: A list containing tuples each containing the benchmark name and a list of sizes.

    Returns:
        A generator of quantum circuits.
    """
    for benchmark, sizes in benchmarks_and_sizes:
        for size in sizes:
            yield get_benchmark(benchmark=benchmark, circuit_size=size, level=BenchmarkLevel.INDEP)


def find_optimized_budgets(
    total_budget: float,
    num_iterations: int,
    counts: LogicalCounts,
) -> tuple[list[float], int, int]:
    """Finds an optimized distribution of error budgets.

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

    default_physical_qubits = default_result["physicalCounts"]["physicalQubits"]
    default_runtime = default_result["physicalCounts"]["runtime"]
    default_metric = default_runtime * default_physical_qubits

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

        physical_qubits = result["physicalCounts"]["physicalQubits"]
        runtime = result["physicalCounts"]["runtime"]

        current_metric = runtime * physical_qubits

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


def _circuit_generator(
    path: str | Path | None = None,
    benchmarks_and_sizes: list[tuple[str, list[int]]] | None = None,
) -> Iterator[QuantumCircuit]:
    """Creates a generator that yields quantum circuits from various sources."""
    if path:
        path = Path(path)
        if path.is_dir():
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".qasm"):
                        yield qasm2.load(
                            str(Path(root) / file),
                            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                        )
        elif path.is_file() and path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                for file_info in zf.infolist():
                    if (
                        file_info.filename.endswith(".qasm")
                        and not file_info.is_dir()
                        and not file_info.filename.startswith("__MACOSX/")
                    ):
                        with zf.open(file_info) as qasm_file:
                            qasm_str = qasm_file.read().decode("utf-8")
                            qc = qasm2.loads(
                                qasm_str,
                                custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
                            )
                            qc.name = Path(file_info.filename).stem
                            yield qc
        else:
            msg = f"Path '{path}' is not a valid directory or .zip file."
            raise ValueError(msg)

    elif benchmarks_and_sizes:
        yield from generate_benchmarks(benchmarks_and_sizes)
    else:
        msg = "Either 'path' or 'benchmarks_and_sizes' must be provided."
        raise ValueError(msg)


def generate_data(
    total_error_budget: float,
    number_of_randomly_generated_distributions: int,
    path: str | Path | None = None,
    benchmarks_and_sizes: list[tuple[str, list[int]]] | None = None,
    logical_counts: list[OrderedDict[str, int]] | None = None,
) -> list[OrderedDict[str, float | int]]:
    """Generates a dataset consisting of logical counts of quantum circuits and respective optimized error budgets.

    This function searches for QASM files in the given directory, loads each circuit,
    estimates its logical counts, and computes optimized error budget partitions using random sampling.
    For each circuit, it collects relevant counts and the optimal error budget distribution,
    appending the results to a list.

    Args:
        total_error_budget: The total error budget to be distributed among error types.
        number_of_randomly_generated_distributions: The number of random distributions to try.
        path: Path to a directory with .qasm files or a .zip file containing them.
        benchmarks_and_sizes: A list of benchmark specifications to generate circuits on the fly.
        logical_counts: A list of logical counts dictionaries to process directly.

    Returns:
        A list of lists, where each inner list contains circuit-specific counts and the corresponding
        optimized error budget partition.
    """
    # Ensure that exactly one of 'path', 'benchmarks_and_sizes', or 'logical_counts' is provided
    provided = [path is not None, benchmarks_and_sizes is not None, logical_counts is not None]
    if sum(provided) != 1:
        msg = "Provide exactly one of 'path', 'benchmarks_and_sizes', or 'logical_counts'."
        raise ValueError(msg)

    if path or benchmarks_and_sizes:
        results = []
        circuit_iterator = _circuit_generator(path, benchmarks_and_sizes)

        for qc in circuit_iterator:
            qc.remove_final_measurements()  # Remove final measurements to avoid estimation errors
            transpiled_qc = transpile(qc, basis_gates=QISKIT_STD_GATES, optimization_level=1)
            try:
                # Estimate logical counts
                counts = estimate(transpiled_qc)["logicalCounts"]
                if counts["rotationCount"] == 0:
                    continue  # Skip circuits without rotations, as we want to ensure distributing error budgets among all three types.
                # Optimize error budgets
                counts = LogicalCounts(counts)
                combinations, *_ = find_optimized_budgets(
                    total_error_budget, number_of_randomly_generated_distributions, counts
                )

                # Collect results
                specific_data = OrderedDict(counts)
                specific_data.update({
                    "logical": combinations[0],
                    "t_states": combinations[1],
                    "rotations": combinations[2],
                })
                results.append(specific_data)
            except Exception as e:
                print(f"Error processing circuit {qc.name}: {e}")

    elif logical_counts:
        results = []
        for c in logical_counts:
            counts = LogicalCounts(c)
            try:
                if counts["rotationCount"] == 0:
                    continue  # Skip circuits without rotations, as we want to ensure distributing error budgets among all three types.
                # Optimize error budgets
                combinations, *_ = find_optimized_budgets(
                    total_error_budget, number_of_randomly_generated_distributions, counts
                )

                # Collect results
                specific_data = OrderedDict(counts)
                specific_data.update({
                    "logical": combinations[0],
                    "t_states": combinations[1],
                    "rotations": combinations[2],
                })
                results.append(specific_data)
            except Exception as e:
                print(f"Error processing logical counts entry {c}: {e}")

    return results
