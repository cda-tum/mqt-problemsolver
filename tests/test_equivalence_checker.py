"""Test the equivalence_checker.py module."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from mqt.problemsolver.equivalence_checker import equivalence_checker

if TYPE_CHECKING:
    import py

alphabet = list(string.ascii_lowercase)


def test_create_condition_string() -> None:
    """Test the function create_condition_string."""
    num_bits = 3
    num_counter_examples = 2
    res_string, counter_examples = equivalence_checker.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    assert isinstance(res_string, str)
    assert isinstance(counter_examples, list)
    assert len(res_string) == 26
    assert len(counter_examples) == num_counter_examples
    assert res_string == "~a & ~b & ~c | a & ~b & ~c"


def test_run_paramter_combinations() -> None:
    """Test the function run_parameter_combinations."""
    num_bits = 6
    num_counter_examples = 3
    res_string, counter_examples = equivalence_checker.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    shots = 512
    delta = 0.7
    result = equivalence_checker.run_parameter_combinations(
        miter=res_string, counter_examples=counter_examples, num_bits=num_bits, shots=shots, delta=delta
    )
    assert result == 5


def test_try_parameter_combinations(tmpdir: py.path.local) -> None:
    """Test the function try_parameter_combinations."""
    p = tmpdir.mkdir("sub")
    equivalence_checker.try_parameter_combinations(
        path=(p / "test.csv"),
        range_deltas=[0.7, 0.8],
        range_num_bits=[5],
        range_fraction_counter_examples=[0.1, 0.2],
        num_runs=5,
    )
    assert len(tmpdir.listdir()) == 1


def test_find_counter_examples() -> None:
    """Test the function find_counter_examples."""
    num_bits = 8
    num_counter_examples = 10
    res_string, counter_examples = equivalence_checker.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    shots = 512
    delta = 0.7
    found_counter_examples = equivalence_checker.find_counter_examples(
        miter=res_string, num_bits=num_bits, shots=shots, delta=delta
    )
    found_counter_examples.sort()
    counter_examples.sort()
    assert found_counter_examples == counter_examples
