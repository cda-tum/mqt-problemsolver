"""Test the equivalence_checker.py module."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from mqt.problemsolver.equivalence_checking import equivalence_checking

if TYPE_CHECKING:
    from pathlib import Path


def test_create_condition_string() -> None:
    """Test the function create_condition_string."""
    num_bits = 3
    num_counter_examples = 2
    res_string, counter_examples = equivalence_checking.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    assert len(res_string) == 26
    assert len(counter_examples) == num_counter_examples
    assert res_string == "~a & ~b & ~c | a & ~b & ~c"

    with pytest.raises(ValueError, match="The number of bits or counter examples cannot be used."):
        equivalence_checking.create_condition_string(num_bits=-5, num_counter_examples=-2)


@pytest.mark.skipif(not importlib.util.find_spec("tweedledum"), reason="tweedledum is not installed")
def test_try_parameter_combinations(tmp_path: Path) -> None:
    """Test the function try_parameter_combinations."""
    path_to_csv = tmp_path / "sub" / "test1.csv"
    path_to_csv.parent.mkdir(parents=True, exist_ok=True)
    equivalence_checking.try_parameter_combinations(
        path=path_to_csv,
        range_deltas=[0.7, 0.8],
        range_num_bits=[5],
        range_fraction_counter_examples=[0.00, 0.05, 0.10, 0.20],
        shots_factor=8,
        num_runs=5,
    )
    assert len(list(tmp_path.iterdir())) == 1
    path_to_csv.unlink()
    path_to_csv.parent.rmdir()


@pytest.mark.skipif(not importlib.util.find_spec("tweedledum"), reason="tweedledum is not installed")
def test_find_counter_examples() -> None:
    """Test the function find_counter_examples."""
    num_bits = 6
    num_counter_examples = 3
    res_string, predetermined_counter_examples = equivalence_checking.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    shots = 512
    delta = 0.7
    found_counter_examples = equivalence_checking.find_counter_examples(
        miter=res_string, num_bits=num_bits, shots=shots, delta=delta
    )
    if isinstance(found_counter_examples, list):
        found_counter_examples.sort()
    predetermined_counter_examples.sort()
    assert found_counter_examples == predetermined_counter_examples

    number_iterations = equivalence_checking.find_counter_examples(
        miter=res_string,
        num_bits=num_bits,
        shots=shots,
        delta=delta,
        predetermined_counter_examples=predetermined_counter_examples,
    )

    # 5 iterations were determined by multiple experiments in the case of 6 bits and 3 counter examples
    assert number_iterations == 5

    # Test, if the correct group of states gets identified as counter examples (using > 50 % counter examples)
    num_bits = 6
    num_counter_examples = 57
    res_string, predetermined_counter_examples = equivalence_checking.create_condition_string(
        num_bits=num_bits, num_counter_examples=num_counter_examples
    )
    shots = 512
    delta = 0.7
    found_counter_examples = equivalence_checking.find_counter_examples(
        miter=res_string, num_bits=num_bits, shots=shots, delta=delta
    )
    if isinstance(found_counter_examples, list):
        found_counter_examples.sort()
    predetermined_counter_examples.sort()
    assert found_counter_examples == predetermined_counter_examples

    with pytest.raises(ValueError, match="Invalid value for delta 1.2, which must be between 0 and 1."):
        equivalence_checking.find_counter_examples(miter=res_string, num_bits=5, shots=shots, delta=1.2)
