import functools
from typing import Any
import numpy as np
import sympy as sp
from arithmetic import ArithmeticItem, SympyTransformer, BreakDownSumsTransformer, Variable

class QUBOGenerator:
    objective_function: ArithmeticItem

    penalties: list[tuple[ArithmeticItem, int | None]]

    def __init__(self, objective_function: ArithmeticItem) -> None:
        self.objective_function = objective_function
        self.penalties = []

    def add_penalty(self, penalty_function: ArithmeticItem, lam: int | None = None) -> None:
        self.penalties.append((penalty_function, lam))

    def construct(self) -> ArithmeticItem:
        return functools.reduce(lambda current, new: current + new[1] * new[0],
                        self._select_lambdas(),
                        self.objective_function)

    def construct_expansion(self) -> sp.Expr:
        expression = self.construct()
        expression = BreakDownSumsTransformer().transform(expression)
        expression = self._construct_expansion(expression)
        return SympyTransformer().transform(expression).expand()

    def _construct_expansion(self, expression: ArithmeticItem) -> ArithmeticItem:
        return expression

    def construct_qubo_matrix(self) -> np.mat:
        coefficients = dict(self.construct_expansion().expand().as_coefficients_dict())
        result = np.zeros((self.get_qubit_count(), self.get_qubit_count()))

        for (var1, i) in self._get_all_variables():
            v1 = sp.Symbol(var1)
            for (var2, j) in self._get_all_variables():
                v2 = sp.Symbol(var2)
                coeff = coefficients.get(v1 * v2, 0)
                if var1 == var2:
                    coeff += coefficients.get(v1, 0)
                if i <= j:
                    result[i - 1][j - 1] += coeff
                else:
                    result[j - 1][i - 1] += coeff

        return result

    def _get_all_variables(self) -> list[tuple[str, int]]:
        return []

    def _select_lambdas(self) -> list[tuple[ArithmeticItem, int]]:
        return self.penalties

    def get_qubit_count(self) -> int:
        return len(self._get_all_variables())

    def get_variable_index(self, _variable: Variable) -> int:
        return 1

    def decode_bit_array(self, _array: list[int]) -> Any:
        return ""
