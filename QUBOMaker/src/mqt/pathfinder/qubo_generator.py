from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.typing as npt
import sympy as sp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

if TYPE_CHECKING:
    from qiskit.opflow import OperatorBase


class QUBOGenerator:
    objective_function: sp.Expr

    penalties: list[tuple[sp.Expr, float | None]]

    def __init__(self, objective_function: sp.Expr) -> None:
        self.objective_function = objective_function
        self.penalties = []

    def add_penalty(self, penalty_function: sp.Expr, lam: int | None = None) -> None:
        self.penalties.append((penalty_function, lam))

    def construct(self) -> sp.Expr:
        return functools.reduce(
            lambda current, new: current + new[1] * new[0],
            self._select_lambdas(),
            self.objective_function,
        )

    def construct_expansion(self) -> sp.Expr:
        expression = self.construct().expand().doit()
        if isinstance(expression, sp.Expr):
            expression = self._construct_expansion(expression).expand()
        expression = expression.doit()
        if isinstance(expression, sp.Expr):
            return expression
        msg = "Expression is not an expression."
        raise TypeError(msg)

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        return expression

    def construct_qubo_matrix(self) -> npt.NDArray[np.int_ | np.float64]:
        coefficients = dict(self.construct_expansion().expand().as_coefficients_dict())
        result = np.zeros((self.get_qubit_count(), self.get_qubit_count()))

        for var1, i in self._get_all_variables():
            for var2, j in self._get_all_variables():
                coeff = coefficients.get(var1 * var2, 0)
                if var1 == var2:
                    coeff += coefficients.get(var1, 0)
                if i <= j:
                    result[i - 1][j - 1] += coeff
                else:
                    result[j - 1][i - 1] += coeff

        return result

    def _get_all_variables(self) -> Sequence[tuple[sp.Expr, int]]:
        return []

    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        return [(expr, weight) if weight is not None else (expr, 1.0) for (expr, weight) in self.penalties]

    def get_qubit_count(self) -> int:
        return len(self._get_all_variables())

    def get_variable_index(self, _variable: sp.Function) -> int:
        return 1

    def decode_bit_array(self, _array: list[int]) -> Any:
        return ""

    def construct_operator(self) -> OperatorBase:
        qubo = self.construct_qubo_matrix()
        quadratic_task = QuadraticProgram()
        quadratic_task.binary_var_list(len(qubo))
        quadratic_task.minimize(quadratic=qubo)
        q = QuadraticProgramToQubo().convert(quadratic_task)
        operator, _ = q.to_ising()
        return operator
