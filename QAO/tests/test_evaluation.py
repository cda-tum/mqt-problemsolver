"""File for making some tests during the Development"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

# for managing symbols
from qubovert import PUBO, boolean_var
from sympy import Expr

from mqt.qao import Constraints, ObjectiveFunction, Problem, Solution, Solver, Variables

# from dwave.cloud import Client


def test_example() -> None:
    """Expected type of final code:
    Problem = Problem
    variables = Problem.variables.add_....._variables(.....)
    def f(var):
       ....
    Problem.objective_function.add_objective_function(f(var))
    Problem.constraints.add_constraints(....)
    # With an eventual option for saving more information (Problem final size, etc.)
    decoded_best_solution, Energy = Problem.solve()
    # And eventually for knowing something about resolution statistics
    Problem.statistics()

    """
    Variables()
    assert 1 == 1


def test_binary_only() -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variables_array("A", [2])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    assert post_dict == {"a": (boolean_var("b0"),), "A_0": (boolean_var("b1"),), "A_1": (boolean_var("b2"),)}


def test_spin_only() -> None:
    """Test only the construction of spin variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_spin_variable("a")
    variables.add_spin_variables_array("A", [2])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    assert post_dict == {
        "a": (boolean_var("b0"), 2, -1),
        "A_0": (boolean_var("b1"), 2, -1),
        "A_1": (boolean_var("b2"), 2, -1),
    }


def test_discrete_only() -> None:
    """Test only the construction of discrete variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_discrete_variable("a", [-1, 1, 3])
    variables.add_discrete_variables_array("A", [2], [-1, 1, 3])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    assert post_dict == {
        "a": ["dictionary", (boolean_var("b0"), -1), (boolean_var("b1"), 1), (boolean_var("b2"), 3)],
        "A_0": ["dictionary", (boolean_var("b3"), -1), (boolean_var("b4"), 1), (boolean_var("b5"), 3)],
        "A_1": ["dictionary", (boolean_var("b6"), -1), (boolean_var("b7"), 1), (boolean_var("b8"), 3)],
    }


@pytest.mark.parametrize(
    ("encoding", "distribution", "precision", "min_val", "max_val"),
    [
        ("dictionary", "uniform", 0.5, -1, 2),
        ("unitary", "uniform", 0.5, -1, 0),
        ("domain well", "uniform", 0.5, -1, 1),
        ("logarithmic 2", "uniform", -1, -1, 2),
        ("arithmetic progression", "uniform", 0.5, -1, 2),
        ("bounded coefficient 1", "uniform", 0.5, -1, 2),
        ("bounded coefficient 8", "uniform", 1, 0, 12),
        ("", "", 0.25, -2, 2),
        ("", "", 0.2, -2, 2),
    ],
)
def test_continuous_only(encoding: str, distribution: str, precision: float, min_val: float, max_val: float) -> None:
    """Test only the construction of continuous variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_continuous_variable("a", min_val, max_val, precision, distribution, encoding)
    variables.add_continuous_variables_array("A", [2], min_val, max_val, precision, distribution, encoding)
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    if (encoding, distribution, precision, min_val, max_val) == ("dictionary", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "dictionary",
                (boolean_var("b0"), -1.0),
                (boolean_var("b1"), -0.5),
                (boolean_var("b2"), 0.0),
                (boolean_var("b3"), 0.5),
                (boolean_var("b4"), 1.0),
                (boolean_var("b5"), 1.5),
                (boolean_var("b6"), 2.0),
            ],
            "A_0": [
                "dictionary",
                (boolean_var("b7"), -1.0),
                (boolean_var("b8"), -0.5),
                (boolean_var("b9"), 0.0),
                (boolean_var("b10"), 0.5),
                (boolean_var("b11"), 1.0),
                (boolean_var("b12"), 1.5),
                (boolean_var("b13"), 2.0),
            ],
            "A_1": [
                "dictionary",
                (boolean_var("b14"), -1.0),
                (boolean_var("b15"), -0.5),
                (boolean_var("b16"), 0.0),
                (boolean_var("b17"), 0.5),
                (boolean_var("b18"), 1.0),
                (boolean_var("b19"), 1.5),
                (boolean_var("b20"), 2.0),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("unitary", "uniform", 0.5, -1, 0):
        assert post_dict == {
            "a": ["unitary", (boolean_var("b0"), 0.5, -1), (boolean_var("b1"), 1)],
            "A_0": ["unitary", (boolean_var("b2"), 0.5, -1), (boolean_var("b3"), 1)],
            "A_1": ["unitary", (boolean_var("b4"), 0.5, -1), (boolean_var("b5"), 1)],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("domain well", "uniform", 0.5, -1, 1):
        assert post_dict == {
            "a": [
                "domain well",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1.5),
                (boolean_var("b3"), 2),
            ],
            "A_0": [
                "domain well",
                (boolean_var("b4"), 0.5, -1),
                (boolean_var("b5"), 1),
                (boolean_var("b6"), 1.5),
                (boolean_var("b7"), 2),
            ],
            "A_1": [
                "domain well",
                (boolean_var("b8"), 0.5, -1),
                (boolean_var("b9"), 1),
                (boolean_var("b10"), 1.5),
                (boolean_var("b11"), 2),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("logarithmic 2", "uniform", -1, -1, 2):
        assert post_dict == {
            "a": ["logarithmic", (boolean_var("b0"), 0.5, -1), (boolean_var("b1"), 1), (boolean_var("b2"), 1.5)],
            "A_0": ["logarithmic", (boolean_var("b3"), 0.5, -1), (boolean_var("b4"), 1), (boolean_var("b5"), 1.5)],
            "A_1": ["logarithmic", (boolean_var("b6"), 0.5, -1), (boolean_var("b7"), 1), (boolean_var("b8"), 1.5)],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("arithmetic progression", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "arithmetic progression",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1.5),
            ],
            "A_0": [
                "arithmetic progression",
                (boolean_var("b3"), 0.5, -1),
                (boolean_var("b4"), 1),
                (boolean_var("b5"), 1.5),
            ],
            "A_1": [
                "arithmetic progression",
                (boolean_var("b6"), 0.5, -1),
                (boolean_var("b7"), 1),
                (boolean_var("b8"), 1.5),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("bounded coefficient 1", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "bounded coefficient",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1),
                (boolean_var("b3"), 1),
            ],
            "A_0": [
                "bounded coefficient",
                (boolean_var("b4"), 0.5, -1),
                (boolean_var("b5"), 1),
                (boolean_var("b6"), 1),
                (boolean_var("b7"), 1),
            ],
            "A_1": [
                "bounded coefficient",
                (boolean_var("b8"), 0.5, -1),
                (boolean_var("b9"), 1),
                (boolean_var("b10"), 1),
                (boolean_var("b11"), 1),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("bounded coefficient 8", "uniform", 1, 0, 12):
        assert post_dict == {
            "a": [
                "logarithmic",
                (boolean_var("b0"), 1, 0),
                (boolean_var("b1"), 2),
                (boolean_var("b2"), 4),
                (boolean_var("b3"), 5),
            ],
            "A_0": [
                "logarithmic",
                (boolean_var("b4"), 1, 0),
                (boolean_var("b5"), 2),
                (boolean_var("b6"), 4),
                (boolean_var("b7"), 5),
            ],
            "A_1": [
                "logarithmic",
                (boolean_var("b8"), 1, 0),
                (boolean_var("b9"), 2),
                (boolean_var("b10"), 4),
                (boolean_var("b11"), 5),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("", "", 0.25, -2, 2):
        assert post_dict == {
            "a": [
                "logarithmic",
                (boolean_var("b0"), 0.25, -2),
                (boolean_var("b1"), 0.5),
                (boolean_var("b2"), 1),
                (boolean_var("b3"), 2),
                (boolean_var("b4"), 0.25),
            ],
            "A_0": [
                "logarithmic",
                (boolean_var("b5"), 0.25, -2),
                (boolean_var("b6"), 0.5),
                (boolean_var("b7"), 1),
                (boolean_var("b8"), 2),
                (boolean_var("b9"), 0.25),
            ],
            "A_1": [
                "logarithmic",
                (boolean_var("b10"), 0.25, -2),
                (boolean_var("b11"), 0.5),
                (boolean_var("b12"), 1),
                (boolean_var("b13"), 2),
                (boolean_var("b14"), 0.25),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == (
        "",
        "",
        0.2,
        -2,
        2,
    ):  # 0.6000000000000001 instead of 0.6 is for a python numerical error
        assert post_dict == {
            "a": [
                "arithmetic progression",
                (boolean_var("b0"), 0.2, -2),
                (boolean_var("b1"), 0.4),
                (boolean_var("b2"), 0.6000000000000001),
                (boolean_var("b3"), 0.8),
                (boolean_var("b4"), 1),
                (boolean_var("b5"), 1),
            ],
            "A_0": [
                "arithmetic progression",
                (boolean_var("b6"), 0.2, -2),
                (boolean_var("b7"), 0.4),
                (boolean_var("b8"), 0.6000000000000001),
                (boolean_var("b9"), 0.8),
                (boolean_var("b10"), 1),
                (boolean_var("b11"), 1),
            ],
            "A_1": [
                "arithmetic progression",
                (boolean_var("b12"), 0.2, -2),
                (boolean_var("b13"), 0.4),
                (boolean_var("b14"), 0.6000000000000001),
                (boolean_var("b15"), 0.8),
                (boolean_var("b16"), 1),
                (boolean_var("b17"), 1),
            ],
        }


def test_cost_function() -> None:
    """Test for cost function translation"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    variables.move_to_binary(constraint.constraints)
    qubo = PUBO()
    qubo = objective_function.rewrite_cost_functions(qubo, variables)
    reference_qubo_dict = {
        ("b0",): 1.0,
        ("b1",): 2.0,
        ("b1", "b4"): -0.25,
        ("b1", "b5"): -0.5,
        ("b1", "b6"): -1.0,
        ("b1", "b7"): -2.0,
        ("b1", "b8"): -0.25,
        ("b2", "b4"): 0.25,
        ("b2",): -2.0,
        ("b2", "b5"): 0.5,
        ("b2", "b6"): 1.0,
        ("b2", "b7"): 2.0,
        ("b2", "b8"): 0.25,
        ("b3", "b4"): 0.75,
        ("b3",): -6.0,
        ("b3", "b5"): 1.5,
        ("b3", "b6"): 3.0,
        ("b3", "b7"): 6.0,
        ("b3", "b8"): 0.75,
        ("b4",): -0.9375,
        ("b5",): -1.75,
        ("b6",): -3.0,
        ("b7",): -4.0,
        ("b8",): -0.9375,
        ("b4", "b5"): 0.25,
        ("b4", "b6"): 0.5,
        ("b4", "b7"): 1.0,
        ("b4", "b8"): 0.125,
        ("b5", "b6"): 1.0,
        ("b5", "b7"): 2.0,
        ("b5", "b8"): 0.25,
        ("b6", "b7"): 4.0,
        ("b6", "b8"): 0.5,
        ("b7", "b8"): 1.0,
        (): 4.0,
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key in reference_qubo_dict:
        reference_qubo_dict_re[tuple(sorted(key))] = reference_qubo_dict[key]
    assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


def test_cost_function_matrix() -> None:
    """Test for cost function translation"""
    variables = Variables()
    constraint = Constraints()
    m1 = variables.add_continuous_variables_array("M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2")
    m2 = variables.add_continuous_variables_array("M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2")
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(np.matmul(m1, m2).item(0, 0))
    variables.move_to_binary(constraint.constraints)
    qubo = PUBO()
    qubo = objective_function.rewrite_cost_functions(qubo, variables)
    reference_qubo_dict = {
        ("b0", "b6"): 0.25,
        ("b0",): -0.5,
        ("b0", "b7"): 0.5,
        ("b0", "b8"): 0.75,
        ("b6",): -0.5,
        ("b7",): -1.0,
        ("b8",): -1.5,
        ("b1", "b6"): 0.5,
        ("b1",): -1.0,
        ("b1", "b7"): 1.0,
        ("b1", "b8"): 1.5,
        ("b2", "b6"): 0.75,
        ("b2",): -1.5,
        ("b2", "b7"): 1.5,
        ("b2", "b8"): 2.25,
        ("b3", "b9"): 0.25,
        ("b3",): -0.5,
        ("b10", "b3"): 0.5,
        ("b11", "b3"): 0.75,
        ("b9",): -0.5,
        ("b10",): -1.0,
        ("b11",): -1.5,
        ("b4", "b9"): 0.5,
        ("b4",): -1.0,
        ("b10", "b4"): 1.0,
        ("b11", "b4"): 1.5,
        ("b5", "b9"): 0.75,
        ("b5",): -1.5,
        ("b10", "b5"): 1.5,
        ("b11", "b5"): 2.25,
        (): 2.0,
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key in reference_qubo_dict:
        reference_qubo_dict_re[tuple(sorted(key))] = reference_qubo_dict[key]
    assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


@pytest.mark.parametrize(
    ("Expression", "var_precision"),
    [
        ("~a = b", False),
        ("a & b = c", False),
        ("a | b = c", False),
        ("a ^ b = c", False),
        ("a + b >= 1", False),
        ("e >= 1", False),
        ("e >= 1", True),
        ("a + b <= 1", False),
        ("a + b > 1", False),
        ("a + b < 1", False),
        ("e <= 1", False),
        ("e <= -1", True),
        ("e > 1", True),
        ("e < 1", False),
        ("e > 1", False),
        ("d < 1", True),
    ],
)
def test_constraint(expression: str, var_precision: bool) -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variable("b")
    variables.add_binary_variable("c")
    variables.add_discrete_variable("d", [-1, 1, 3])
    variables.add_continuous_variable("e", -2, 2, 0.25, "", "")
    variables.move_to_binary(constraint.constraints)
    constraint.add_constraint(expression, True, True, var_precision)
    constraint.translate_constraints(variables)
    dictionary_constraints_qubo = {
        ("b3",): -1.0,
        ("b4",): -1.0,
        ("b5",): -1.0,
        ("b3", "b4"): 2.0,
        ("b3", "b5"): 2.0,
        ("b4", "b5"): 2.0,
        (): 1.0,
    }
    qubo_first = constraint.constraints_penalty_functions[0][0]
    qubo_second = constraint.constraints_penalty_functions[1][0]

    qubo_first_re = {}
    for key in qubo_first:
        qubo_first_re[tuple(sorted(key))] = qubo_first[key]
    qubo_second_re = {}
    for key in qubo_second:
        qubo_second_re[tuple(sorted(key))] = qubo_second[key]
    dictionary_constraints_qubo_re = {}
    for key in dictionary_constraints_qubo:
        dictionary_constraints_qubo_re[tuple(sorted(key))] = dictionary_constraints_qubo[key]
    if expression == "~a = b":
        dictionary_constraints_qubo_2 = {("b0",): -1.0, ("b1",): -1.0, ("b0", "b1"): 2.0, (): 1.0}
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a & b = c":
        dictionary_constraints_qubo_2 = {("b0", "b1"): 1.0, ("b0", "b2"): -2.0, ("b1", "b2"): -2.0, ("b2",): 3.0}
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a | b = c":
        dictionary_constraints_qubo_2 = {
            ("b0", "b1"): 1.0,
            ("b0",): 1.0,
            ("b1",): 1.0,
            ("b0", "b2"): -2.0,
            ("b1", "b2"): -2.0,
            ("b2",): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a ^ b = c":
        dictionary_constraints_qubo_2 = {
            ("b0", "b1", "b2"): 4.0,
            ("b0",): 1,
            ("b1",): 1.0,
            ("b2",): 1.0,
            ("b0", "b1"): -2.0,
            ("b0", "b2"): -2.0,
            ("b1", "b2"): -2.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b >= 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -1.0,
            ("b1",): -1.0,
            ("__a0",): 3.0,
            ("b0", "b1"): 2.0,
            ("__a0", "b0"): -2.0,
            ("__a0", "b1"): -2.0,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e >= 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): 7.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.5,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -1.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -2.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -4.0,
            ("__a0", "b10"): -0.5,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e >= 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): 1.5625,
            ("__a1",): 3.25,
            ("__a2",): 1.5625,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.125,
            ("__a1", "b6"): -0.25,
            ("__a2", "b6"): -0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -0.25,
            ("__a1", "b7"): -0.5,
            ("__a2", "b7"): -0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -0.5,
            ("__a1", "b8"): -1.0,
            ("__a2", "b8"): -0.5,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -1.0,
            ("__a1", "b9"): -2.0,
            ("__a2", "b9"): -1.0,
            ("__a0", "b10"): -0.125,
            ("__a1", "b10"): -0.25,
            ("__a2", "b10"): -0.125,
            ("__a0", "__a1"): 0.25,
            ("__a0", "__a2"): 0.125,
            ("__a1", "__a2"): 0.25,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b <= 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -1.0,
            ("b1",): -1.0,
            ("__a0",): -1.0,
            ("b0", "b1"): 2.0,
            ("__a0", "b0"): 2.0,
            ("__a0", "b1"): 2.0,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b > 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -3.0,
            ("b1",): -3.0,
            ("b0", "b1"): 2.0,
            (): 4.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b < 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): 1.0,
            ("b1",): 1.0,
            ("b0", "b1"): 2.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e <= 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): -5.0,
            ("__a1",): -8.0,
            ("__a0", "__a1"): 4.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.5,
            ("__a1", "b6"): 1.0,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 1.0,
            ("__a1", "b7"): 2.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 2.0,
            ("__a1", "b8"): 4.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): 4.0,
            ("__a1", "b9"): 8.0,
            ("__a0", "b10"): 0.5,
            ("__a1", "b10"): 1.0,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e <= -1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -0.4375,
            ("b7",): -0.75,
            ("b8",): -1.0,
            ("b10",): -0.4375,
            ("__a0",): -0.4375,
            ("__a1",): -0.75,
            ("__a2",): -0.4375,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.125,
            ("__a1", "b6"): 0.25,
            ("__a2", "b6"): 0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 0.25,
            ("__a1", "b7"): 0.5,
            ("__a2", "b7"): 0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 0.5,
            ("__a1", "b8"): 1.0,
            ("__a2", "b8"): 0.5,
            ("b10", "b9"): 1.0,
            ("__a0", "b9"): 1.0,
            ("__a1", "b9"): 2.0,
            ("__a2", "b9"): 1.0,
            ("__a0", "b10"): 0.125,
            ("__a1", "b10"): 0.25,
            ("__a2", "b10"): 0.125,
            ("__a0", "__a1"): 0.25,
            ("__a0", "__a2"): 0.125,
            ("__a1", "__a2"): 0.25,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e > 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.9375,
            ("b7",): -3.75,
            ("b8",): -7.0,
            ("b9",): -12.0,
            ("b10",): -1.9375,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("b10", "b9"): 1,
            (): 16.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e < 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -0.9375,
            ("b7",): -1.75,
            ("b8",): -3.0,
            ("b9",): -4.0,
            ("b10",): -0.9375,
            ("__a0",): -3.0,
            ("__a1",): -3.0,
            ("__a0", "__a1"): 2.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.5,
            ("__a1", "b6"): 0.5,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 1.0,
            ("__a1", "b7"): 1.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 2.0,
            ("__a1", "b8"): 2.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): 4.0,
            ("__a1", "b9"): 4.0,
            ("__a0", "b10"): 0.5,
            ("__a1", "b10"): 0.5,
            (): 4.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e > 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.5625,
            ("b7",): -3.0,
            ("b8",): -5.5,
            ("b9",): -9.0,
            ("b10",): -1.5625,
            ("__a0",): 1.6875,
            ("__a1",): 3.5,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.125,
            ("__a1", "b6"): -0.25,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -0.25,
            ("__a1", "b7"): -0.5,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -0.5,
            ("__a1", "b8"): -1.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -1.0,
            ("__a1", "b9"): -2.0,
            ("__a0", "b10"): -0.125,
            ("__a1", "b10"): -0.25,
            ("__a0", "__a1"): 0.25,
            (): 10.5625,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "d < 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b3",): 1.0,
            ("b4",): 1.0,
            ("b5",): 9.0,
            ("__a0",): 1.0,
            ("b3", "b4"): -2.0,
            ("b3", "b5"): -6.0,
            ("__a0", "b3"): -2.0,
            ("b4", "b5"): 6.0,
            ("__a0", "b4"): 2.0,
            ("__a0", "b5"): 6.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))


@pytest.mark.parametrize(
    "Expression",
    ["~b0 = b1"],
)
def test_constraint_no_sub(expression: str) -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variable("b")
    variables.add_binary_variable("c")
    variables.add_discrete_variable("d", [-1, 1, 3])
    variables.add_continuous_variable("e", -2, 2, 0.25, "", "")
    variables.move_to_binary(constraint.constraints)
    constraint.add_constraint(expression, True, False)
    constraint.translate_constraints(variables)
    dictionary_constraints_qubo = {
        ("b3",): -1.0,
        ("b4",): -1.0,
        ("b5",): -1.0,
        ("b3", "b4"): 2.0,
        ("b3", "b5"): 2.0,
        ("b4", "b5"): 2.0,
        (): 1.0,
    }
    qubo_first = constraint.constraints_penalty_functions[0][0]
    qubo_second = constraint.constraints_penalty_functions[1][0]
    qubo_first_re = {}
    for key in qubo_first:
        qubo_first_re[tuple(sorted(key))] = qubo_first[key]
    qubo_second_re = {}
    for key in qubo_second:
        qubo_second_re[tuple(sorted(key))] = qubo_second[key]
    dictionary_constraints_qubo_re = {}
    for key in dictionary_constraints_qubo:
        dictionary_constraints_qubo_re[tuple(sorted(key))] = dictionary_constraints_qubo[key]
    if expression == "~b0 = b1":
        dictionary_constraints_qubo_2 = {("b0",): -1, ("b1",): -1, ("b0", "b1"): 2, (): 1}
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
        "maximum_coefficient",
        "VLM",
        "MOMC",
        "MOC",
        "upper lower bound naive",
        "upper lower bound posiform and negaform method",
    ],
)
def test_problem(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    constraint.add_constraint("c >= 1", True, True, False)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    qubo = problem.write_the_final_cost_function(lambda_strategy)
    lambdas_or = problem.lambdas
    lambdas = [1.1 * el for el in lambdas_or]
    reference_qubo_dict = {
        ("b0",): 1.0,
        ("b1",): 2.0 - lambdas[1],
        ("b1", "b2"): 2.0 * lambdas[1],
        ("b1", "b3"): 2.0 * lambdas[1],
        ("b1", "b4"): -0.25,
        ("b1", "b5"): -0.5,
        ("b1", "b6"): -1.0,
        ("b1", "b7"): -2.0,
        ("b1", "b8"): -0.25,
        ("b2", "b4"): 0.25,
        ("b2",): -2.0 - lambdas[1],
        ("b2", "b3"): 2.0 * lambdas[1],
        ("b2", "b5"): 0.5,
        ("b2", "b6"): 1.0,
        ("b2", "b7"): 2.0,
        ("b2", "b8"): 0.25,
        ("b3", "b4"): 0.75,
        ("b3",): -6.0 - lambdas[1],
        ("b3", "b5"): 1.5,
        ("b3", "b6"): 3.0,
        ("b3", "b7"): 6.0,
        ("b3", "b8"): 0.75,
        ("b4",): -0.9375 - 1.4375 * lambdas[0],
        ("b5",): -1.75 - 2.75 * lambdas[0],
        ("b6",): -3.0 - 5.0 * lambdas[0],
        ("b7",): -4.0 - 8.0 * lambdas[0],
        ("b8",): -0.9375 - 1.4375 * lambdas[0],
        ("__a0",): 7.0 * lambdas[0],
        ("b4", "b5"): 0.25 + 0.25 * lambdas[0],
        ("b4", "b6"): 0.5 + 0.5 * lambdas[0],
        ("b4", "b7"): 1.0 + 1.0 * lambdas[0],
        ("b4", "b8"): 0.125 + 0.125 * lambdas[0],
        ("__a0", "b4"): -0.5 * lambdas[0],
        ("b5", "b6"): 1.0 + 1.0 * lambdas[0],
        ("b5", "b7"): 2.0 + 2.0 * lambdas[0],
        ("b5", "b8"): 0.25 + 0.25 * lambdas[0],
        ("__a0", "b5"): -1.0 * lambdas[0],
        ("b6", "b7"): 4.0 + 4.0 * lambdas[0],
        ("b6", "b8"): 0.5 + 0.5 * lambdas[0],
        ("__a0", "b6"): -2.0 * lambdas[0],
        ("b7", "b8"): 1.0 + 1.0 * lambdas[0],
        ("__a0", "b7"): -4.0 * lambdas[0],
        ("__a0", "b8"): -0.5 * lambdas[0],
        (): 4.0 + lambdas[1] + 9.0 * lambdas[0],
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key in reference_qubo_dict:
        reference_qubo_dict_re[tuple(sorted(key))] = reference_qubo_dict[key]
    if lambda_strategy == "upper_bound_only_positive" or lambda_strategy == "upper lower bound naive":
        assert [52.25 * 1.1] * 2 == lambdas
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "maximum_coefficient":
        assert [10.0 * 1.1] * 2 == lambdas
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "VLM" or lambda_strategy == "MOMC":
        assert [12.0 * 1.1] * 2 == lambdas
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "MOC":
        assert [7 * 1.1, 6 * 1.1] == lambdas
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "upper lower bound posiform and negaform method":
        assert [31.625 * 1.1] * 2 == lambdas
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
        "maximum_coefficient",
        "VLM",
        "MOMC",
        "MOC",
        "upper lower bound naive",
        "upper lower bound posiform and negaform method",
    ],
)
def test_simulated_annealer_solver(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)
    if isinstance(solution, Solution):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.5}
        assert solution.best_energy < -2.24  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -2.26
        assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.25}
        assert all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "constraint_expr"),
    [
        ("upper_bound_only_positive", "c >= 1"),
        ("maximum_coefficient", "c >= 1"),
        ("VLM", "c >= 1"),
        ("MOMC", "c >= 1"),
        ("MOC", "c >= 1"),
        ("upper lower bound naive", "c >= 1"),
        ("upper lower bound posiform and negaform method", "c >= 1"),
        ("maximum_coefficient", "c > 1"),
        ("VLM", "c > 1"),
        ("MOMC", "c > 1"),
        ("MOC", "c > 1"),
        ("upper lower bound naive", "c > 1"),
        ("upper lower bound posiform and negaform method", "c > 1"),
        ("maximum_coefficient", "b < 1"),
        ("VLM", "b < 1"),
        ("MOMC", "b < 1"),
        ("MOC", "b < 1"),
        ("upper lower bound naive", "b < 1"),
        ("upper lower bound posiform and negaform method", "b < 1"),
        ("maximum_coefficient", "b <= 1"),
        ("VLM", "b <= 1"),
        ("MOMC", "b <= 1"),
        ("MOC", "b <= 1"),
        ("upper lower bound naive", "b <= 1"),
        ("upper lower bound posiform and negaform method", "b <= 1"),
        ("upper_bound_only_positive", "b + c >= 2"),
        ("maximum_coefficient", "b + c >= 2"),
        ("VLM", "b + c >= 2"),
        ("MOMC", "b + c >= 2"),
        ("MOC", "b + c >= 2"),
        ("upper lower bound naive", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "b + c >= 2"),
        ("upper_bound_only_positive", "a = 1"),
        ("maximum_coefficient", "a = 1"),
        ("VLM", "a = 1"),
        ("MOMC", "a = 1"),
        ("MOC", "a = 1"),
        ("upper lower bound naive", "a = 1"),
        ("upper lower bound posiform and negaform method", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained(lambda_strategy: str, constraint_expr: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.4 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution == {"a": 0.0, "b": 1.0, "c": -0.5}
                or solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5}
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_update", "constraint_expr"),
    [
        ("sequential penalty increase", "c >= 1"),
        ("scaled sequential penalty increase", "c >= 1"),
        ("binary search penalty algorithm", "c >= 1"),
        ("sequential penalty increase", "c > 1"),
        ("scaled sequential penalty increase", "c > 1"),
        ("binary search penalty algorithm", "c > 1"),
        ("sequential penalty increase", "b < 1"),
        ("scaled sequential penalty increase", "b < 1"),
        ("binary search penalty algorithm", "b < 1"),
        ("sequential penalty increase", "b <= 1"),
        ("scaled sequential penalty increase", "b <= 1"),
        ("binary search penalty algorithm", "b <= 1"),
        ("sequential penalty increase", "b + c >= 2"),
        ("scaled sequential penalty increase", "b + c >= 2"),
        ("binary search penalty algorithm", "b + c >= 2"),
        ("sequential penalty increase", "a = 1"),
        ("scaled sequential penalty increase", "a = 1"),
        ("binary search penalty algorithm", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained_lambda_update_mechanism(
    lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy="manual", lambda_value=5.0
    )
    solver.get_lambda_updates()
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution == {"a": 0.0, "b": 1.0, "c": -0.5}
                or solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5}
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "c >= 1"),
        ("VLM", "sequential penalty increase", "c >= 1"),
        ("MOMC", "sequential penalty increase", "c >= 1"),
        ("MOC", "sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c >= 1"),
        ("VLM", "scaled sequential penalty increase", "c >= 1"),
        ("MOMC", "scaled sequential penalty increase", "c >= 1"),
        ("MOC", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c >= 1"),
        ("VLM", "binary search penalty algorithm", "c >= 1"),
        ("MOMC", "binary search penalty algorithm", "c >= 1"),
        ("MOC", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c >= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "sequential penalty increase", "c > 1"),
        ("VLM", "sequential penalty increase", "c > 1"),
        ("MOMC", "sequential penalty increase", "c > 1"),
        ("MOC", "sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c > 1"),
        ("VLM", "scaled sequential penalty increase", "c > 1"),
        ("MOMC", "scaled sequential penalty increase", "c > 1"),
        ("MOC", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c > 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c > 1"),
        ("VLM", "binary search penalty algorithm", "c > 1"),
        ("MOMC", "binary search penalty algorithm", "c > 1"),
        ("MOC", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c > 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "sequential penalty increase", "b < 1"),
        ("VLM", "sequential penalty increase", "b < 1"),
        ("MOMC", "sequential penalty increase", "b < 1"),
        ("MOC", "sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b < 1"),
        ("VLM", "scaled sequential penalty increase", "b < 1"),
        ("MOMC", "scaled sequential penalty increase", "b < 1"),
        ("MOC", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b < 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b < 1"),
        ("VLM", "binary search penalty algorithm", "b < 1"),
        ("MOMC", "binary search penalty algorithm", "b < 1"),
        ("MOC", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b < 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "sequential penalty increase", "b <= 1"),
        ("VLM", "sequential penalty increase", "b <= 1"),
        ("MOMC", "sequential penalty increase", "b <= 1"),
        ("MOC", "sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b <= 1"),
        ("VLM", "scaled sequential penalty increase", "b <= 1"),
        ("MOMC", "scaled sequential penalty increase", "b <= 1"),
        ("MOC", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b <= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b <= 1"),
        ("VLM", "binary search penalty algorithm", "b <= 1"),
        ("MOMC", "binary search penalty algorithm", "b <= 1"),
        ("MOC", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b <= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "sequential penalty increase", "b + c >= 2"),
        ("VLM", "sequential penalty increase", "b + c >= 2"),
        ("MOMC", "sequential penalty increase", "b + c >= 2"),
        ("MOC", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b + c >= 2"),
        ("VLM", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOMC", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOC", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b + c >= 2"),
        ("maximum_coefficient", "binary search penalty algorithm", "b + c >= 2"),
        ("VLM", "binary search penalty algorithm", "b + c >= 2"),
        ("MOMC", "binary search penalty algorithm", "b + c >= 2"),
        ("MOC", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound naive", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b + c >= 2"),
        ("upper_bound_only_positive", "sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "sequential penalty increase", "a = 1"),
        ("VLM", "sequential penalty increase", "a = 1"),
        ("MOMC", "sequential penalty increase", "a = 1"),
        ("MOC", "sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "a = 1"),
        ("VLM", "scaled sequential penalty increase", "a = 1"),
        ("MOMC", "scaled sequential penalty increase", "a = 1"),
        ("MOC", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "a = 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "a = 1"),
        ("VLM", "binary search penalty algorithm", "a = 1"),
        ("MOMC", "binary search penalty algorithm", "a = 1"),
        ("MOC", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained_lambda_update_mechanism_and_strategy(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy=lambda_strategy
    )
    solver.get_lambda_updates()
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution == {"a": 0.0, "b": 1.0, "c": -0.5}
                or solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5}
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "M1_0_1 >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "M1_0_1 >= 1"),
        ("VLM", "sequential penalty increase", "M1_0_1 >= 1"),
        ("MOMC", "sequential penalty increase", "M1_0_1 >= 1"),
        ("MOC", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("VLM", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("MOMC", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("MOC", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("VLM", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("MOMC", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("MOC", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "M1_0_1 >= 1"),
    ],
)
def test_simulated_annealing_cost_function_matrix(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for cost function translation"""
    variables = Variables()
    m1 = variables.add_continuous_variables_array("M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2")
    m2 = variables.add_continuous_variables_array("M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2")
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(np.matmul(m1, m2).item(0, 0))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy=lambda_strategy
    )
    solver.get_lambda_updates()
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "M1_0_1 >= 1":
            assert (
                solution.best_solution == {"M1": [[-1, 2]], "M2": [[2], [-1]]}
                or solution.best_solution == {"M1": [[2, 2]], "M2": [[-1], [-1]]}
                or not all_satisfy
            )
            assert (solution.best_energy < -3.9) or not all_satisfy
            assert solution.best_energy > -4.1 or not all_satisfy
            assert (
                solution.optimal_solution_cost_functions_values() == {"M1_0_0*M2_0_0 + M1_0_1*M2_1_0": -4.0}
                or not all_satisfy
            )


'''
@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "c >= 1"),
        ("VLM", "sequential penalty increase", "c >= 1"),
        ("MOMC", "sequential penalty increase", "c >= 1"),
        ("MOC", "sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c >= 1"),
        ("VLM", "scaled sequential penalty increase", "c >= 1"),
        ("MOMC", "scaled sequential penalty increase", "c >= 1"),
        ("MOC", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c >= 1"),
        ("VLM", "binary search penalty algorithm", "c >= 1"),
        ("MOMC", "binary search penalty algorithm", "c >= 1"),
        ("MOC", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c >= 1"),
    ],
)
def test_quantum_annealer_solver_constrained_lambda_update_mechanism_and_strategy(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    print("Quantum test\n")
    variables = Variables()
    token=
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_Dwave_quantum_annealer(
        problem, token=token, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy=lambda_strategy, annealing_time_scheduling=10.0, num_reads=1000
    )
    lambda_updates = solver.get_lambda_updates()
    print(lambda_updates)
    print(constraint_expr)
    print(lambda_strategy)
    print(lambda_update)
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        print(each_satisfy)
        print(solution.best_solution)
        print(solution.optimal_solution_cost_functions_values())
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
    else:
        assert solution
'''


def test_gas_solver_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_grover_adaptive_search_qubo(problem, qubit_values=6, num_runs=10)
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


def test_qaoa_solver_qubo_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_qaoa_qubo(
        problem,
        num_runs=10,
    )
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


def test_vqe_solver_qubo_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_vqe_qubo(
        problem,
        num_runs=10,
    )
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
    ],
)
def test_qaoa_solver(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_qaoa_qubo(problem, num_runs=10, lambda_strategy=lambda_strategy)
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.5}
        assert solution.best_energy < -2.24  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -2.26
        assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.25}
        assert all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
    ],
)
def test_vqe_solver(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_vqe_qubo(problem, num_runs=10, lambda_strategy=lambda_strategy)
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.5}
        assert solution.best_energy < -2.24  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -2.26
        assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.25}
        assert all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "c >= 1"),
        ("VLM", "sequential penalty increase", "c >= 1"),
        ("MOMC", "sequential penalty increase", "c >= 1"),
        ("MOC", "sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c >= 1"),
        ("VLM", "scaled sequential penalty increase", "c >= 1"),
        ("MOMC", "scaled sequential penalty increase", "c >= 1"),
        ("MOC", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c >= 1"),
        ("VLM", "binary search penalty algorithm", "c >= 1"),
        ("MOMC", "binary search penalty algorithm", "c >= 1"),
        ("MOC", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c >= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "sequential penalty increase", "c > 1"),
        ("VLM", "sequential penalty increase", "c > 1"),
        ("MOMC", "sequential penalty increase", "c > 1"),
        ("MOC", "sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c > 1"),
        ("VLM", "scaled sequential penalty increase", "c > 1"),
        ("MOMC", "scaled sequential penalty increase", "c > 1"),
        ("MOC", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c > 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c > 1"),
        ("VLM", "binary search penalty algorithm", "c > 1"),
        ("MOMC", "binary search penalty algorithm", "c > 1"),
        ("MOC", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c > 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "sequential penalty increase", "b < 1"),
        ("VLM", "sequential penalty increase", "b < 1"),
        ("MOMC", "sequential penalty increase", "b < 1"),
        ("MOC", "sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b < 1"),
        ("VLM", "scaled sequential penalty increase", "b < 1"),
        ("MOMC", "scaled sequential penalty increase", "b < 1"),
        ("MOC", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b < 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b < 1"),
        ("VLM", "binary search penalty algorithm", "b < 1"),
        ("MOMC", "binary search penalty algorithm", "b < 1"),
        ("MOC", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b < 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "sequential penalty increase", "b <= 1"),
        ("VLM", "sequential penalty increase", "b <= 1"),
        ("MOMC", "sequential penalty increase", "b <= 1"),
        ("MOC", "sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b <= 1"),
        ("VLM", "scaled sequential penalty increase", "b <= 1"),
        ("MOMC", "scaled sequential penalty increase", "b <= 1"),
        ("MOC", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b <= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b <= 1"),
        ("VLM", "binary search penalty algorithm", "b <= 1"),
        ("MOMC", "binary search penalty algorithm", "b <= 1"),
        ("MOC", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b <= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "sequential penalty increase", "b + c >= 2"),
        ("VLM", "sequential penalty increase", "b + c >= 2"),
        ("MOMC", "sequential penalty increase", "b + c >= 2"),
        ("MOC", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b + c >= 2"),
        ("VLM", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOMC", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOC", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b + c >= 2"),
        ("maximum_coefficient", "binary search penalty algorithm", "b + c >= 2"),
        ("VLM", "binary search penalty algorithm", "b + c >= 2"),
        ("MOMC", "binary search penalty algorithm", "b + c >= 2"),
        ("MOC", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound naive", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b + c >= 2"),
        ("upper_bound_only_positive", "sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "sequential penalty increase", "a = 1"),
        ("VLM", "sequential penalty increase", "a = 1"),
        ("MOMC", "sequential penalty increase", "a = 1"),
        ("MOC", "sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "a = 1"),
        ("VLM", "scaled sequential penalty increase", "a = 1"),
        ("MOMC", "scaled sequential penalty increase", "a = 1"),
        ("MOC", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "a = 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "a = 1"),
        ("VLM", "binary search penalty algorithm", "a = 1"),
        ("MOMC", "binary search penalty algorithm", "a = 1"),
        ("MOC", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "a = 1"),
    ],
)
def test_qaoa_constrained_lambda_update_mechanism_and_strategy(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_qaoa_qubo(
        problem,
        num_runs=10,
        max_lambda_update=10,
        lambda_update_mechanism=lambda_update,
        lambda_strategy=lambda_strategy,
    )
    solver.get_lambda_updates()
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution == {"a": 0.0, "b": 1.0, "c": -0.5}
                or solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5}
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "c >= 1"),
        ("VLM", "sequential penalty increase", "c >= 1"),
        ("MOMC", "sequential penalty increase", "c >= 1"),
        ("MOC", "sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c >= 1"),
        ("VLM", "scaled sequential penalty increase", "c >= 1"),
        ("MOMC", "scaled sequential penalty increase", "c >= 1"),
        ("MOC", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c >= 1"),
        ("VLM", "binary search penalty algorithm", "c >= 1"),
        ("MOMC", "binary search penalty algorithm", "c >= 1"),
        ("MOC", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c >= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "sequential penalty increase", "c > 1"),
        ("VLM", "sequential penalty increase", "c > 1"),
        ("MOMC", "sequential penalty increase", "c > 1"),
        ("MOC", "sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c > 1"),
        ("VLM", "scaled sequential penalty increase", "c > 1"),
        ("MOMC", "scaled sequential penalty increase", "c > 1"),
        ("MOC", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c > 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c > 1"),
        ("VLM", "binary search penalty algorithm", "c > 1"),
        ("MOMC", "binary search penalty algorithm", "c > 1"),
        ("MOC", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c > 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "sequential penalty increase", "b < 1"),
        ("VLM", "sequential penalty increase", "b < 1"),
        ("MOMC", "sequential penalty increase", "b < 1"),
        ("MOC", "sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b < 1"),
        ("VLM", "scaled sequential penalty increase", "b < 1"),
        ("MOMC", "scaled sequential penalty increase", "b < 1"),
        ("MOC", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b < 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b < 1"),
        ("VLM", "binary search penalty algorithm", "b < 1"),
        ("MOMC", "binary search penalty algorithm", "b < 1"),
        ("MOC", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b < 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "sequential penalty increase", "b <= 1"),
        ("VLM", "sequential penalty increase", "b <= 1"),
        ("MOMC", "sequential penalty increase", "b <= 1"),
        ("MOC", "sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b <= 1"),
        ("VLM", "scaled sequential penalty increase", "b <= 1"),
        ("MOMC", "scaled sequential penalty increase", "b <= 1"),
        ("MOC", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b <= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b <= 1"),
        ("VLM", "binary search penalty algorithm", "b <= 1"),
        ("MOMC", "binary search penalty algorithm", "b <= 1"),
        ("MOC", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b <= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "sequential penalty increase", "b + c >= 2"),
        ("VLM", "sequential penalty increase", "b + c >= 2"),
        ("MOMC", "sequential penalty increase", "b + c >= 2"),
        ("MOC", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b + c >= 2"),
        ("VLM", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOMC", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOC", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b + c >= 2"),
        ("maximum_coefficient", "binary search penalty algorithm", "b + c >= 2"),
        ("VLM", "binary search penalty algorithm", "b + c >= 2"),
        ("MOMC", "binary search penalty algorithm", "b + c >= 2"),
        ("MOC", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound naive", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b + c >= 2"),
        ("upper_bound_only_positive", "sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "sequential penalty increase", "a = 1"),
        ("VLM", "sequential penalty increase", "a = 1"),
        ("MOMC", "sequential penalty increase", "a = 1"),
        ("MOC", "sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "a = 1"),
        ("VLM", "scaled sequential penalty increase", "a = 1"),
        ("MOMC", "scaled sequential penalty increase", "a = 1"),
        ("MOC", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "a = 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "a = 1"),
        ("VLM", "binary search penalty algorithm", "a = 1"),
        ("MOMC", "binary search penalty algorithm", "a = 1"),
        ("MOC", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "a = 1"),
    ],
)
def test_vqe_constrained_lambda_update_mechanism_and_strategy(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_vqe_qubo(
        problem,
        num_runs=10,
        max_lambda_update=10,
        lambda_update_mechanism=lambda_update,
        lambda_strategy=lambda_strategy,
    )
    solver.get_lambda_updates()
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution == {"a": 0.0, "b": 1.0, "c": -0.5}
                or solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5}
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
    ],
)
def test_gas_solver(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    # a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 3])
    c0 = variables.add_continuous_variable("c", -1, 1, 0.5, "", "")
    cost_function = cast(Expr, b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_grover_adaptive_search_qubo(
        problem, qubit_values=7, num_runs=10, coeff_precision=0.5, lambda_strategy=lambda_strategy
    )
    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"b": 3.0, "c": -1.0}
        assert solution.best_energy < -1.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -2.1
        assert solution.optimal_solution_cost_functions_values() == {"b*c + c**2": -2.0}
        assert all_satisfy
    else:
        assert solution
