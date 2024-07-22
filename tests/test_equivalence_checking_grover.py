from __future__ import annotations

import string

from mqt.problemsolver.equivalence_checker import executer

alphabet = list(string.ascii_lowercase)


def create_condition_string(num_qubits: int, num_targets: int) -> tuple[str, list[str]]:
    """
    Creates a string to simulate a miter out of bitstring combinations (e.g. '0000' -> 'a & b & c & d')

    Parameters
    ----------
    num_qubits : int
        Number of input bits
    num_targets : int
        Number of counter examples

    Returns
    -------
    res_string : str
        Resulting condition string
    list_of_bitstrings : list[str]
        The corresponding bitstrings to res_string (e.g. list_of_bitstrings is ['0000'] for res_string 'a & b & c & d')
    """

    if num_qubits < 0 or num_targets < 0:
        raise TypeError

    list_of_bitstrings: list[str] = []
    if num_targets == 0:
        res, _ = create_condition_string(num_qubits, 1)
        res += " & a"
        return res, list_of_bitstrings
    res_string: str = ""
    list_of_bitstrings = []
    for num in range(num_targets):
        bitstring = list(str(format(num, f"0{num_qubits}b")))
        list_of_bitstrings.append(str(format(num, f"0{num_qubits}b")))
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
        if num < num_targets - 1:
            res_string += combined_bitstring + " | "
        else:
            res_string += combined_bitstring
    return res_string, list_of_bitstrings


def test_create_condition_string() -> None:
    num_qubits = 3
    num_targets = 2
    res_string, list_of_bitstrings = create_condition_string(num_qubits=num_qubits, num_targets=num_targets)

    assert isinstance(res_string, str)
    assert isinstance(list_of_bitstrings, list)
    assert len(res_string) == 26
    assert len(list_of_bitstrings) == num_targets
    assert res_string == "~a & ~b & ~c | ~a & ~b & c"


def test_run() -> None:
    num_qubits = 3
    num_targets = 2
    shots = 128
    delta = 0.7
    number_of_processes = 8
    miter, solutions = create_condition_string(num_qubits, num_targets)
    res_states: dict[int, list[str]] = executer.find_counter_examples(
        miter, num_qubits, shots, delta, number_of_processes
    )
    for process in res_states:
        print(type(res_states))
        assert sorted(process) == sorted(solutions)

    num_qubits = 6
    num_targets = 10
    shots = 512
    delta = 0.8
    number_of_processes = 4
    miter, solutions = create_condition_string(num_qubits, num_targets)
    res_states: dict[int, list[str]] = executer.find_counter_examples(
        miter, num_qubits, shots, delta, number_of_processes
    )
    for process in res_states:
        assert sorted(process) == sorted(solutions)


test_run()
