from __future__ import annotations

import string

import numpy as np

import pandas as pd

# from mqt.problemsolver.equivalence_checker import sampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, PhaseOracle
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

sim_counts = AerSimulator(method="statevector")

alphabet = list(string.ascii_lowercase)


def create_condition_string(num_bits: int, num_counter_examples: int) -> tuple[str, list[str]]:
    """
    Creates a string to simulate a miter out of bitstring combinations (e.g. '0000' -> 'a & b & c & d')

    Parameters
    ----------
    num_bits : int
        Number of input bits
    num_counter_examples : int
        Number of counter examples

    Returns
    -------
    res_string : str
        Resulting condition string
    counter_examples : list[str]
        The corresponding bitstrings to res_string (e.g. counter_examples is ['0000'] for res_string 'a & b & c & d')
    """

    if num_bits < 0 or num_counter_examples < 0:
        raise ValueError

    counter_examples: list[str] = []
    if num_counter_examples == 0:
        res, _ = create_condition_string(num_bits, 1)
        res += " & a"
        return res, counter_examples
    res_string: str = ""
    counter_examples = []
    for num in range(num_counter_examples):
        bitstring = list(str(format(num, f"0{num_bits}b")))[::-1]
        counter_examples.append(str(format(num, f"0{num_bits}b")))
        for i, char in enumerate(bitstring):
            if char == "0" and i == 0:
                bitstring[i] = "~" + alphabet[i]
            elif char == "1" and i == 0:
                bitstring[i] = alphabet[i]
            elif char == "0":
                bitstring[i] = " & " + "~" + alphabet[i]
            elif char == "1":
                bitstring[i] = " & " + alphabet[i]
        combined_bitstring = "".join(bitstring)
        if num < num_counter_examples - 1:
            res_string += combined_bitstring + " | "
        else:
            res_string += combined_bitstring
    return res_string, counter_examples


def sampler(
    miter: str,
    counter_examples: list[str],
    start_iterations: int,
    shots: int,
    delta: float,
) -> str:
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
        num_bits = operator.num_qubits
        total_num_combinations = 2**num_bits

        qc = QuantumCircuit(num_bits)
        qc.h(list(range(num_bits)))
        qc.compose(operator.power(iterations).decompose(), inplace=True)
        qc.measure_all()

        qc = transpile(qc, sim_counts)

        job = sim_counts.run(qc, shots=shots)
        result = job.result()
        counts_dict = dict(result.get_counts())
        counts_list = list(counts_dict.values())
        counts_list.sort(reverse=True)

        counts_dict = dict(
            sorted(counts_dict.items(), key=lambda item: item[1])[::-1]
        )  # Sort state dictionary with respect to values (counts)

        target_states = []
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

    # if sorted(target_states) == sorted(counter_examples):
    #     return f"Correct targets found! Total number of iterations: {total_iterations}"
    # if len(target_states) == 0:
    #     if len(counter_examples) > 0:
    #         return f"No targets found! Total number of iterations: {total_iterations}"
    #     if len(counter_examples) == 0:
    #         return f"Correct targets found (None)! Total number of iterations: {total_iterations}"

    # return f"At least one wrong target found! Total number of iterations: {total_iterations}"

    if sorted(target_states) == sorted(counter_examples):
        return total_iterations
    if len(target_states) == 0:
        if len(counter_examples) > 0:
            return None
        if len(counter_examples) == 0:
            return total_iterations

    return None


def find_counter_examples(
    miter: str,
    counter_examples: list[str],
    num_bits: int,
    shots: int,
    delta: float,
) -> str:
    """
    Runs the grover verification application in multiple processes.

    Parameters
    ----------
    miter: str
        String that contains the conditions to satisfy
    num_bits : int
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
        print(f"Invalid delta of {delta}. It must be between 0 and 1.")

    total_num_combinations = 2**num_bits
    start_iterations = np.floor(np.pi / (4 * np.arcsin((1 / total_num_combinations) ** 0.5)) - 0.5).astype(int)
    return sampler(miter, counter_examples, start_iterations, shots, delta)


def run(path, range_deltas, range_num_bits, range_fraction_counter_examples, num_runs, verbose=False):
    data = pd.DataFrame(columns = ['Input Bits','Counter Examples'] + [delta for delta in deltas])
    i = 0
    for num_bits in range_num_bits:
        for fraction_counter_examples in range_fraction_counter_examples:
            row = [num_bits, fraction_counter_examples]
            for delta in range_deltas:
                if verbose:
                    print(f"num_bits: {num_bits}, fraction_counter_examples: {fraction_counter_examples}, delta: {delta}")
                results = []
                for run in range(num_runs):
                    num_counter_examples = round(fraction_counter_examples * 2**num_bits)
                    miter, counter_examples = create_condition_string(num_bits, num_counter_examples)
                    result = find_counter_examples(miter, counter_examples, num_bits, 8 * (2**num_bits), delta)
                    results.append(result)
                if None in results:
                    row.append("-")
                else:
                    row.append(np.mean(results))
            data.loc[i] = row
            i += 1

    data.to_csv(path, index=False)
