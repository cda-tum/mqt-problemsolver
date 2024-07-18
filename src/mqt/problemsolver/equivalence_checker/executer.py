from __future__ import annotations

import multiprocessing
import string

import numpy as np

# from mqt.problemsolver.equivalence_checker import sampler

from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, PhaseOracle
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

sim_counts = AerSimulator(method="statevector")


def sampler(
    process_number: int,
    return_dict: dict[int, str | int],
    miter: str,
    start_iterations: int,
    shots: int,
    delta: float,
) -> None:
    """
    Runs the algorithm utilizing Grover's algorithm to look for elements satisfying the conditions.

    Parameters
    ----------
    process_number : int
        Index of the process
    return_dict : dict[int, str | int]
        Dictionary to share results with parent process
    miter: str
        String that contains the conditions to satisfy
    start_iterations: int
        Defines the number of Grover iterations to start with
    shots: int
        Number of shots the algorithm should run for
    delta: float
        Threshold parameter between 0 and 1
    """
    total_iterations = 0
    for iterations in reversed(range(1, start_iterations + 1)):
        total_iterations += iterations
        oracle = PhaseOracle(miter)

        operator = GroverOperator(oracle).decompose()
        num_qubits = operator.num_qubits
        total_num_combinations = 2**num_qubits

        qc = QuantumCircuit(num_qubits)
        qc.h(list(range(num_qubits)))
        qc.compose(operator.power(iterations).decompose(), inplace=True)
        qc.measure_all()

        qc = transpile(qc, sim_counts)

        job = sim_counts.run(qc, shots=shots)
        result = job.result()
        counts_dict = dict(result.get_counts())
        counts_list = list(counts_dict.values())
        counts_list.sort(reverse=True)

        counts_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1])[::-1]) # Sort state dictionary with respect to values (counts)

        stopping_condition = False
        for i in range(round(total_num_combinations * 0.5)):
            if (i + 1) == len(counts_list):
                stopping_condition = True
                targets_list = counts_list
                targets_dict = {
                    list(counts_dict.keys())[t]: list(counts_dict.values())[t] for t in range(len(targets_list))
                }
                target_states = list(targets_dict.keys())
                break

            diff = counts_list[i] - counts_list[i + 1]
            if diff > counts_list[i] * delta:
                stopping_condition = True
                targets_list = counts_list[: i + 1]
                targets_dict = {
                    list(counts_dict.keys())[t]: list(counts_dict.values())[t] for t in range(len(targets_list))
                }
                target_states = list(targets_dict.keys())
                break

        if stopping_condition:
            break
 
    for i, state in enumerate(target_states):
        target_states[i] = state[::-1] # Compensate Qiskit's qubit ordering

    return_dict[process_number] = target_states


def run(miter: str, num_qubits:int, shots: int, delta: float, number_of_processes: int) -> list[str | int]:
    """
    Runs the grover verification application in multiple processes.

    Parameters
    ----------
    miter: str
        String that contains the conditions to satisfy
    num_qubits : int
        Number of input bits
    shots: int
        Number of shots
    delta: float
        Threshold parameter between 0 and 1
    number_of_processes: int
        Number of processes the algorithm should run in simultaneously

    Returns
    -------
    list[str | int]
        A list of values representing the targets found by the Grover algorithm
    """
    try:
        assert 0 <= delta <= 1
    except AssertionError:
        print(f'Invalid delta of {delta}. It must be between 0 and 1.')
    
    total_num_combinations = 2**num_qubits
    start_iterations = np.floor(np.pi / (4 * np.arcsin((1 / total_num_combinations) ** 0.5)) - 0.5).astype(int)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(number_of_processes):
        process = multiprocessing.Process(
            target=sampler,
            args=(i, return_dict, miter, start_iterations, shots, delta),
        )
        jobs.append(process)
        process.start()

    for job in jobs:
        job.join()
    return return_dict.values()

miter = "a & b"
targets = run(miter, 2, 32, 0.7, 1)