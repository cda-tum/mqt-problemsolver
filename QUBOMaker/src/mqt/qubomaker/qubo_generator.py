"""Provides a base class for QUBO generators that can be extended for different problem classes.
"""
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
    """A base class for QUBO generators that can be extended for different problem classes.

    Collects constraints and penalties and provides methods for constructing the QUBO representation of a problem.

    Attributes:
        objective_function (sp.Expr | None): The objective function of the problem.
        penalties (list[tuple[sp.Expr, float | None]]): The constraints and corresponding penalties.
    """

    objective_function: sp.Expr | None

    penalties: list[tuple[sp.Expr, float | None]]

    expansion_cache: sp.Expr | None = None

    def __init__(self, objective_function: sp.Expr | None) -> None:
        """Initializes a new QUBOGenerator instance.

        Args:
            objective_function (sp.Expr | None): The objective function to be used by the QUBO generator.
        """
        self.objective_function = objective_function
        self.penalties = []

    def add_penalty(self, penalty_function: sp.Expr, lam: int | None = None) -> None:
        """Adds a cost function for a constraint to the problem instance.

        A penalty factor can be specified to scale the penalty function. Otherwise, a fitting penalty factor will be
        estimated automatically.

        Args:
            penalty_function (sp.Expr): A cost function that represents a constraint.
            lam (int | None, optional): The penalty scaling factor. Defaults to None.
        """
        self.expansion_cache = None
        self.penalties.append((penalty_function, lam))

    def construct(self) -> sp.Expr:
        """Constructs a mathematical representation of the QUBO formulation.

        This representation is in its simplest form, including sum and product terms.

        Returns:
            sp.Expr: The mathematical representation of the QUBO formulation.
        """
        return cast(
            sp.Expr,
            functools.reduce(
                lambda current, new: current + new[1] * new[0],
                self._select_lambdas(),
                self.objective_function if self.objective_function is not None else 0,
            ),
        )

    def construct_expansion(self) -> sp.Expr:
        """Constructs a mathematical representation of the QUBO formulation and expands it.

        This will expand sum and product terms into full sums and products of each of their elements.
        The final result will be a sum of terms where each term is a product of variables and scalars.

        Raises:
            TypeError: If the constructed QUBO formulation is not a sympy expression.

        Returns:
            sp.Expr: A mathematical representation of the QUBO formulation in expanded form.
        """

        if self.expansion_cache is not None:
            return self.expansion_cache
        expression = self.construct().expand().doit()
        if isinstance(expression, sp.Expr):
            expression = self._construct_expansion(expression).expand()
        expression = expression.doit().expand()
        if isinstance(expression, sp.Expr):
            expression = self.expand_higher_order_terms(expression)
            self.expansion_cache = expression
            return expression
        msg = "Expression is not an expression."
        raise TypeError(msg)

    def expand_higher_order_terms(self, expression: sp.Expr) -> sp.Expr:
        """Expands a mathematical QUBO expression.

        Terms of order 3 or higher will be transformed into quadratic terms by adding auxiliary variables recursively until
        the order is 2.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        result = 0
        auxiliary_dict: dict[sp.Expr, sp.Expr] = {}
        coeffs = expression.as_coefficients_dict()  # type: ignore[no-untyped-call]
        for term in coeffs:
            unpowered = self.__unpower(term)
            unpowered = self.__simplify_auxiliary_variables(unpowered, auxiliary_dict)
            order = self.__get_order(unpowered)
            if order <= 2:
                result += unpowered * coeffs[term]
                continue
            new_term = self.__decrease_order(unpowered, auxiliary_dict)
            result += new_term * coeffs[term]
        return cast(sp.Expr, result)

    def __simplify_auxiliary_variables(self, expression: sp.Expr, auxiliary_dict: dict[sp.Expr, sp.Expr]) -> sp.Expr:
        """Minimizes the number of requires auxiliary variables by removing products that have already been transformed in previous steps.

        Args:
            expression (sp.Expr): The expression to optimize
            auxiliary_dict (dict[sp.Expr, sp.Expr]): A dictionary mapping existing products of variables to their resulting auxiliary variable.

        Returns:
            sp.Expr: The optimized expression.
        """
        if not isinstance(expression, sp.Mul):
            return expression
        used_auxilliaries = {term for term in expression.args if term in auxiliary_dict.values()}
        redundant_variables = {term for term in auxiliary_dict if auxiliary_dict[term] in used_auxilliaries}
        remaining_variables = [arg for arg in expression.args if arg not in redundant_variables]
        if len(remaining_variables) == 1:
            return cast(sp.Expr, remaining_variables[0])
        return cast(sp.Expr, sp.Mul(*remaining_variables))

    def __optimal_decomposition(
        self, terms: tuple[sp.Expr, ...], auxiliary_dict: dict[sp.Expr, sp.Expr]
    ) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
        """Computes the optimal decomposition of a product of variables into terms of order 2.

        Args:
            terms (tuple[sp.Expr, ...]): The terms of the product.
            auxiliary_dict (dict[sp.Expr, sp.Expr]): The previously used auxiliary variables.

        Returns:
            tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]: A tuple containing the two variables that are multiplied, the auxiliary variable used for them, and the remaining expression.
        """
        for x1 in terms:
            for x2 in terms:
                if x1 == x2:
                    continue
                if (x1 * x2) not in auxiliary_dict:
                    continue
                return (x1, x2, auxiliary_dict[x1 * x2], sp.Mul(*[term for term in terms if term not in (x1, x2)]))
        x1 = terms[0]
        x2 = terms[1]
        y: sp.Symbol = sp.Symbol(f"y_{len(auxiliary_dict) + 1}")  # type: ignore[no-untyped-call]
        auxiliary_dict[x1 * x2] = y
        rest = sp.Mul(*terms[2:])
        return (x1, x2, y, rest)

    def __decrease_order(self, expression: sp.Expr, auxiliary_dict: dict[sp.Expr, sp.Expr]) -> sp.Expr:
        """Decreases the order of a product of variables by adding auxiliary variables.

        Args:
            expression (sp.Expr): The expression to transform.
            auxiliary_dict (dict[sp.Expr, sp.Expr]): A dictionary of previously used auxiliary variables.

        Returns:
            sp.Expr: The new expression with lower order.
        """
        (x1, x2, y, rest) = self.__optimal_decomposition(cast(tuple[sp.Expr, ...], expression.args), auxiliary_dict)
        auxiliary_penalty = x1 * x2 - 2 * y * x1 - 2 * y * x2 + 3 * y
        rest = rest * y
        if self.__get_order(rest) > 2:
            rest = self.__decrease_order(rest, auxiliary_dict)
        return cast(sp.Expr, auxiliary_penalty + rest)

    def __get_order(self, expression: sp.Expr) -> int:
        """Computes the order of a product of variables.

        Args:
            expression (sp.Expr): The expression to check.

        Returns:
            int: The order of the expression. If the expression is not a product, 1 is returned.
        """
        if isinstance(expression, sp.Mul):
            return sum([self.__get_order(arg) for arg in expression.args])
        return 1

    def __unpower(self, expression: sp.Expr) -> sp.Expr:
        """Removes exponentiation from an expression.

        This matters, because when using binary variables, x^2 = x always holds. This allows us to compute the order of the
        term more easily.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        if isinstance(expression, sp.Pow):
            return cast(sp.Expr, expression.args[0])
        if isinstance(expression, sp.Mul):
            return cast(sp.Expr, sp.Mul(*[self.__unpower(arg) for arg in expression.args]))
        return expression

    def __get_auxiliary_variables(self, expression: sp.Expr) -> list[sp.Symbol]:
        """Returns a list of all auxiliary variables used in an expression.

        Auxiliary variables will start with "y_" by definition.

        Args:
            expression (sp.Expr): The expression to check.

        Returns:
            list[sp.Symbol]: The list of employed auxiliary variables.
        """
        if isinstance(expression, sp.Mul):
            return list({var for arg in expression.args for var in self.__get_auxiliary_variables(arg)})
        if isinstance(expression, sp.Symbol) and str(expression).startswith("y_"):
            return [expression]
        return []

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        """A method that can be extended by classes that inherit from QUBOGenerator to transform the QUBO formulation into expanded form,
        if that process requires additional steps.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression.
        """
        return expression

    def construct_qubo_matrix(self) -> npt.NDArray[np.int_ | np.float64]:
        """Constructs the matrix representation of the QUBO problem.

        This is achieved by first creating the expanded QUBO formula, and then taking the coefficients of each term.

        Returns:
            npt.NDArray[np.int_ | np.float64]: The matrix representation of the QUBO problem.
        """
        coefficients = dict(self.construct_expansion().expand().as_coefficients_dict())
        auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
        auxiliary_variables.sort(key=lambda var: int(str(var)[2:]))
        result = np.zeros(
            (self.get_variable_count() + len(auxiliary_variables), self.get_variable_count() + len(auxiliary_variables))
        )

        all_variables = dict(self._get_all_variables())
        print(all_variables)
        print(auxiliary_variables)

        def get_index(variable: sp.Expr) -> int:
            if variable in all_variables:
                return all_variables[variable] - 1
            return auxiliary_variables.index(cast(sp.Symbol, variable)) + self.get_variable_count()

        for term in coefficients:
            if isinstance(term, sp.Mul):
                index1 = get_index(term.args[0])
                index2 = get_index(term.args[1])
                if index1 > index2:
                    index1, index2 = index2, index1
                result[index1][index2] = coefficients[term]
            elif isinstance(term, (sp.Symbol, sp.Function)):
                index = get_index(term)
                result[index][index] = coefficients[term]

        return result

    def get_cost(self, assignment: list[int]) -> float:
        """Given an assignment, computes the total cost value of the corresponding cost function evaluation.

        Args:
            assignment (list[int]): The assignment for each variable (either 0 or 1).

        Returns:
            float: The cost value for the assignment.
        """
        expansion = self.construct_expansion()
        variable_assignment = [(item[0], assignment[item[1] - 1]) for item in self._get_all_variables()]
        return cast(float, expansion.subs(variable_assignment).evalf())  # type: ignore[no-untyped-call]

    def _get_all_variables(self) -> Sequence[tuple[sp.Expr, int]]:
        """Returns all non-auxiliary variables used in the QUBO formulation.

        Returns:
            Sequence[tuple[sp.Expr, int]]: A list of tuples containing the variable and its index.
        """
        return []

    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        """Computes the penalty factors for each constraint. May be extended by subclasses.

        Returns:
            list[tuple[sp.Expr, float]]: A list of tuples containing the individual cost functions and their constraints.
        """
        return [(expr, weight) if weight is not None else (expr, 1.0) for (expr, weight) in self.penalties]

    def get_variable_count(self) -> int:
        """Returns the number of binary variables required to represent the QUBO problem.

        Returns:
            int: The number of required binary variables.
        """
        return len(self._get_all_variables())

    def get_variable_index(self, _variable: sp.Function) -> int:
        """For a given variable, returns its index in the QUBO matrix.

        Args:
            _variable (sp.Function): The variable to investigate.

        Returns:
            int: The index of the variable.
        """
        return 1

    def decode_bit_array(self, _array: list[int]) -> Any:
        """Given an assignment, decodes it into a meaningful result. May be extended by subclasses.

        Args:
            _array (list[int]): The binary assignment.

        Returns:
            Any: The decoded result.
        """
        return ""

    def construct_operator(self) -> OperatorBase:
        """Construct the ising operator representing the QUBO problem.

        Returns:
            OperatorBase: The ising operator of this QUBO problem.
        """
        qubo = self.construct_qubo_matrix()
        quadratic_task = QuadraticProgram()
        quadratic_task.binary_var_list(len(qubo))
        quadratic_task.minimize(quadratic=qubo)
        q = QuadraticProgramToQubo().convert(quadratic_task)
        operator, _ = q.to_ising()
        return operator
