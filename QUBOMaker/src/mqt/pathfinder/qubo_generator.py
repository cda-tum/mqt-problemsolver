from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import sympy as sp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

if TYPE_CHECKING:
    from qiskit.opflow import OperatorBase


class QUBOGenerator:
    objective_function: sp.Expr | None

    penalties: list[tuple[sp.Expr, float | None]]

    def __init__(self, objective_function: sp.Expr | None) -> None:
        self.objective_function = objective_function
        self.penalties = []

    def add_penalty(self, penalty_function: sp.Expr, lam: int | None = None) -> None:
        self.penalties.append((penalty_function, lam))

    def construct(self) -> sp.Expr:
        return cast(
            sp.Expr,
            functools.reduce(
                lambda current, new: current + new[1] * new[0],
                self._select_lambdas(),
                self.objective_function if self.objective_function is not None else 0,
            ),
        )

    def construct_expansion(self) -> sp.Expr:
        expression = self.construct().expand().doit()
        if isinstance(expression, sp.Expr):
            expression = self._construct_expansion(expression).expand()
        expression = expression.doit()
        if isinstance(expression, sp.Expr):
            return self.expand_higher_order_terms(expression)
        msg = "Expression is not an expression."
        raise TypeError(msg)

    def expand_higher_order_terms(self, expression: sp.Expr) -> sp.Expr:
        result = 0
        auxilliary_index = 1
        coeffs = expression.as_coefficients_dict()  # type: ignore[no-untyped-call]
        for term in coeffs:
            unpowered = self.__unpower(term)
            order = self.__get_order(unpowered)
            if order <= 2:
                result += unpowered * coeffs[term]
                continue
            (new_term, auxilliary_index) = self.__decrease_order(unpowered, auxilliary_index)
            result += new_term * coeffs[term]
        return cast(sp.Expr, result)

    def __decrease_order(self, expression: sp.Expr, next_auxilliary: int) -> tuple[sp.Expr, int]:
        x1 = cast(sp.Expr, expression.args[0])
        x2 = cast(sp.Expr, expression.args[1])
        y = sp.Symbol(f"y_{next_auxilliary}")  # type: ignore[no-untyped-call]
        auxilliary_penalty = x1 * x2 - 2 * y * x1 - 2 * y * x2 + 3 * y
        auxilliary_index = next_auxilliary + 1
        rest = sp.Mul(*expression.args[2:]) * y
        if self.__get_order(rest) > 2:
            rest, auxilliary_index = self.__decrease_order(rest, auxilliary_index)
        return (auxilliary_penalty + rest, auxilliary_index)

    def __get_order(self, expression: sp.Expr) -> int:
        if isinstance(expression, sp.Mul):
            return sum([self.__get_order(arg) for arg in expression.args])
        return 1

    def __unpower(self, expression: sp.Expr) -> sp.Expr:
        if isinstance(expression, sp.Pow):
            return cast(sp.Expr, expression.args[0])
        if isinstance(expression, sp.Mul):
            return cast(sp.Expr, sp.Mul(*[self.__unpower(arg) for arg in expression.args]))
        return expression

    def __get_auxilliary_variables(self, expression: sp.Expr) -> list[sp.Symbol]:
        if isinstance(expression, sp.Mul):
            return list({var for arg in expression.args for var in self.__get_auxilliary_variables(arg)})
        if isinstance(expression, sp.Symbol) and str(expression).startswith("y_"):
            return [expression]
        return []

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        return expression

    def construct_qubo_matrix(self) -> npt.NDArray[np.int_ | np.float64]:
        coefficients = dict(self.construct_expansion().expand().as_coefficients_dict())
        auxilliary_variables = list({var for arg in coefficients for var in self.__get_auxilliary_variables(arg)})
        auxilliary_variables.sort(key=lambda var: int(str(var)[2:]))
        result = np.zeros(
            (self.get_qubit_count() + len(auxilliary_variables), self.get_qubit_count() + len(auxilliary_variables))
        )

        all_variables = dict(self._get_all_variables())

        def get_index(variable: sp.Expr) -> int:
            if variable in all_variables:
                return all_variables[variable] - 1
            return auxilliary_variables.index(cast(sp.Symbol, variable)) + self.get_qubit_count()

        for term in coefficients:
            if isinstance(term, sp.Mul):
                index1 = get_index(term.args[0])
                index2 = get_index(term.args[1])
                if index1 > index2:
                    index1, index2 = index2, index1
                result[index1][index2] = coefficients[term]
            elif isinstance(term, sp.Symbol):
                index = get_index(term)
                result[index][index] = coefficients[term]

        return result

    def get_cost(self, assignment: list[int]) -> float:
        expansion = self.construct_expansion()
        variable_assignment = [(item[0], assignment[item[1] - 1]) for item in self._get_all_variables()]
        return cast(float, expansion.subs(variable_assignment).evalf())  # type: ignore[no-untyped-call]

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
