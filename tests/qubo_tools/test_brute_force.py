# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module tests the brute-force solver of the QuboGenerator class."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy as sp

from mqt.problemsolver.qubo_tools import QuboGenerator


def test_brute_force_finds_minimum() -> None:
    """Tests that the brute-force solver returns the optimal assignment and its cost, including constant terms."""
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    generator = QuboGenerator(3 * x1 * x2 - 2 * x2 * x3 + 5 * x1 - 4 * x3)
    generator.add_penalty((x1 + x2 - 1) ** 2, 10)

    assignment, cost = generator.solve_brute_force()

    assert assignment == [0, 1, 1]
    assert cost == pytest.approx(-6.0)
    assert cost == pytest.approx(float(generator.get_cost(assignment)))


def test_brute_force_matches_exhaustive_cost_evaluation() -> None:
    """Tests that the brute-force cost equals the minimum of `get_cost` over all assignments."""
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    generator = QuboGenerator(4 * x1 * x2 - 3 * x1 * x3 + x2 * x3 - 2 * x2 + x3 - 1)

    _, cost = generator.solve_brute_force()

    expected = min(float(generator.get_cost(list(bits))) for bits in itertools.product([0, 1], repeat=3))
    assert cost == pytest.approx(expected)


def test_brute_force_with_auxiliary_variables() -> None:
    """Tests that the brute-force solver handles auxiliary variables from higher-order terms."""
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    generator = QuboGenerator(-5 * x1 * x2 * x3 + x1 + x2 + x3)

    assignment, cost = generator.solve_brute_force()

    qubo = generator.construct_qubo_matrix()
    assert len(assignment) == len(qubo)
    assert len(assignment) > generator.get_encoding_variable_count()
    assert assignment[: generator.get_encoding_variable_count()] == [1, 1, 1]
    assert cost == pytest.approx(-2.0)
    x = np.array(assignment)
    assert cost == pytest.approx(float(x @ qubo @ x))
