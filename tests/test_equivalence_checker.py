"""Test the equivalence_checker.py module."""

from __future__ import annotations

import importlib
import string
from typing import TYPE_CHECKING

import pytest

from mqt.problemsolver.equivalence_checker import equivalence_checker

if TYPE_CHECKING:
    from pathlib import Path

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

    with pytest.raises(ValueError, match="The number of bits or counter examples cannot be used."):
        equivalence_checker.create_condition_string(num_bits=-5, num_counter_examples=-2)


@pytest.mark.skipif(not importlib.util.find_spec("tweedledum"), reason="tweedledum is not installed")
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
    with pytest.raises(ValueError, match="Invalid value for delta 1.2, which must be between 0 and 1."):
        equivalence_checker.run_parameter_combinations(
            miter=res_string, counter_examples=counter_examples, num_bits=num_bits, shots=shots, delta=1.2
        )


@pytest.mark.skipif(not importlib.util.find_spec("tweedledum"), reason="tweedledum is not installed")
def test_try_parameter_combinations(tmp_path: Path) -> None:
    """Test the function try_parameter_combinations."""
    d = tmp_path / "sub"
    d.mkdir()
    d = d / "test1.csv"
    d = d.absolute()
    string_path = d.as_posix()
    equivalence_checker.try_parameter_combinations(
        path=string_path,
        range_deltas=[0.7, 0.8],
        range_num_bits=[5],
        range_fraction_counter_examples=[0.00, 0.05, 0.10, 0.20],
        num_runs=5,
    )
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.skipif(not importlib.util.find_spec("tweedledum"), reason="tweedledum is not installed")
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
