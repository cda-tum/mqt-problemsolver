from __future__ import annotations

import string

from mqt.problemsolver.equivalence_checker import executer

alphabet = list(string.ascii_lowercase)


def create_condition_string(num_qubits: int, num_counter_examples: int) -> tuple[str, list[str]]:
    """
    Creates a string to simulate a miter out of bitstring combinations (e.g. '0000' -> 'a & b & c & d')

    Parameters
    ----------
    num_qubits : int
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

    if num_qubits < 0 or num_counter_examples < 0:
        raise ValueError

    counter_examples: list[str] = []
    if num_counter_examples == 0:
        res, _ = create_condition_string(num_qubits, 1)
        res += " & a"
        return res, counter_examples
    res_string: str = ""
    counter_examples = []
    for num in range(num_counter_examples):
        bitstring = list(str(format(num, f"0{num_qubits}b")))[::-1]
        counter_examples.append(str(format(num, f"0{num_qubits}b")))
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


if __name__ == "__main__":

    def test_create_condition_string() -> None:
        num_qubits = 3
        num_counter_examples = 2
        res_string, counter_examples = create_condition_string(
            num_qubits=num_qubits, num_counter_examples=num_counter_examples
        )
        assert isinstance(res_string, str)
        assert isinstance(counter_examples, list)
        assert len(res_string) == 26
        assert len(counter_examples) == num_counter_examples
        assert res_string == "~a & ~b & ~c | a & ~b & ~c"

    def test_run() -> None:
        num_qubits = 4
        num_counter_examples = 2
        shots = 128
        delta = 0.7
        miter, counter_examples = create_condition_string(num_qubits, num_counter_examples)
        result = executer.find_counter_examples(miter, counter_examples, num_qubits, shots, delta)
        assert result == "Correct targets found! Total number of iterations: 2"

        num_qubits = 6
        num_counter_examples = 10
        shots = 512
        delta = 0.8
        miter, counter_examples = create_condition_string(num_qubits, num_counter_examples)
        result = executer.find_counter_examples(miter, counter_examples, num_qubits, shots, delta)
        assert result == "Correct targets found! Total number of iterations: 5"
