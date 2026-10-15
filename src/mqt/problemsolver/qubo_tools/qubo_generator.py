# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Provides a base class for QUBO generators that can be extended for different problem classes."""

from __future__ import annotations

import functools
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import qiskit
import qiskit.circuit
import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt

    from .device import Calibration


class SlackChainAssignment:
    """Stores assignment information for a chain of variables in a QUBO formulation."""

    chains: list[list[str]]
    slack_dict: dict[str, tuple[str, str]]
    indices: dict[str, int]

    def __init__(self) -> None:
        """Creates a new SlackChainAssignment instance."""
        self.chains = []
        self.indices = {}
        self.slack_dict = {}

    def add_slack_variable(self, new_slack: str, replace_1: str, replace_2: str) -> None:
        """Adds a new auxiliary variable to the assignment, replacing the two given existing variables.

        Args:
            new_slack: The new auxiliary variable.
            replace_1: The first variable to be replaced.
            replace_2: The second variable to be replaced.
        """
        self.slack_dict[new_slack] = (replace_1, replace_2)
        parts = new_slack.split("_")
        chain_index = int(parts[1]) - 1
        while chain_index >= len(self.chains):
            self.chains.append([])
        self.chains[chain_index].append(new_slack)

    def compute_indices(self, expr: sp.Expr, offset: int = 0) -> None:
        """Computes the indices of all variables in the assignment based on their names and stores them as a side-effect.

        Args:
            expr: The expression to analyze.
            offset: The starting offset. Defaults to 0.
        """
        symbols = [str(s) for s in expr.free_symbols]
        prime_symbols = [s for s in symbols if s.endswith("'")]
        encoding_symbols = [s for s in symbols if not s.endswith("'") and s.startswith("x_")]
        slack_symbols = [s for s in symbols if not s.endswith("'") and s.startswith("y_")]
        self.indices.clear()
        for enc in encoding_symbols:
            self.indices[enc] = int(enc[2:]) - 1 + offset
        slack_symbols.sort(key=lambda x: tuple(int(i) for i in x[2:].split("_")))
        for slack in slack_symbols:
            self.indices[slack] = len(self.indices)
        prime_symbols.sort(key=lambda x: tuple([x.count("'")] + [int(i) for i in x.replace("'", "")[2:].split("_")]))
        for prime in prime_symbols:
            self.indices[prime] = len(self.indices)


class QuboGenerator:
    """A base class for QUBO generators that can be extended for different problem classes.

    Collects constraints and penalties and provides methods for constructing the QUBO representation of a problem.

    Attributes:
        objective_function: The objective function of the problem.
        penalties: The constraints and corresponding penalties.
    """

    objective_function: sp.Expr | None

    penalties: list[tuple[sp.Expr, float | None]]

    expansion_cache: sp.Expr | None = None

    auxiliary_cache: dict[sp.Expr, sp.Expr]

    disable_caching: bool = (
        False  # If set to True, the caching of the expansion will be disabled. This is useful for debugging.
    )

    def __init__(self, objective_function: sp.Expr | None) -> None:
        """Initializes a new QuboGenerator instance.

        Args:
            objective_function: The objective function to be used by the QUBO generator.
        """
        self.objective_function = objective_function
        self.penalties = []
        self.auxiliary_cache = {}

    def add_penalty(self, penalty_function: sp.Expr, lam: float | None = None) -> None:
        """Adds a cost function for a constraint to the problem instance.

        A penalty factor can be specified to scale the penalty function. Otherwise, a fitting penalty factor will be
        estimated automatically.

        Args:
            penalty_function: A cost function that represents a constraint.
            lam: The penalty scaling factor. Defaults to None.
        """
        self.expansion_cache = None
        self.auxiliary_cache = {}
        self.penalties.append((penalty_function, lam))

    def construct(self) -> sp.Expr:
        """Constructs a mathematical representation of the QUBO formulation.

        This representation is in its simplest form, including sum and product terms.

        Returns:
            The mathematical representation of the QUBO formulation.
        """
        return cast(
            "sp.Expr",
            functools.reduce(
                lambda current, new: current + new[1] * new[0],
                self._select_lambdas(),
                self.objective_function if self.objective_function is not None else 0,
            ),
        )

    def construct_expansion(
        self, include_slack_information: bool = False, for_embedding: bool = False
    ) -> sp.Expr | tuple[sp.Expr, SlackChainAssignment]:
        """Constructs a mathematical representation of the QUBO formulation and expands it.

        This will expand sum and product terms into full sums and products of each of their elements.
        The final result will be a sum of terms where each term is a product of variables and scalars.

        Args:
            include_slack_information: Whether to include information about slack variables in the result. Defaults to False.
            for_embedding: Whether to prepare the QUBO for embedding on a quantum device. Defaults to False.

        Returns:
            A mathematical representation of the QUBO formulation in expanded form.

        Raises:
            TypeError: If the constructed QUBO formulation is not a sympy expression.
        """
        if self.expansion_cache is not None and not include_slack_information and not self.disable_caching:
            return self.expansion_cache
        expression = self.construct().expand().doit().doit()
        if isinstance(expression, sp.Expr):
            expression = self._construct_expansion(expression).expand()
        expression = expression.doit().expand()
        embedded_assignment = SlackChainAssignment()
        if for_embedding:
            unifying_substitution = {var: sp.Symbol(f"x_{i}") for (var, i) in self._get_encoding_variables()}
            expression = expression.subs(unifying_substitution)
            expression = self.expand_higher_order_terms(expression, embedded_assignment)
        else:
            expression = self.expand_higher_order_terms_greedy_minimization(expression, embedded_assignment)
            for symbol in list(expression.free_symbols) + [tup[0] for tup in self._get_encoding_variables()]:
                assert isinstance(symbol, (sp.Expr))
                expression = expression.subs({symbol**2: symbol})
        self.expansion_cache = expression

        if include_slack_information:
            embedded_assignment.compute_indices(expression)
            return expression, embedded_assignment
        return expression

    def expand_higher_order_terms(
        self, expression: sp.Expr, assignment: SlackChainAssignment, chain_index: int = 1
    ) -> sp.Expr:
        """Expands higher order terms in a cost function to reduce the order.

        Args:
            expression: The expression to transform.
            assignment: The assignment to store information about the slack variables.
            chain_index: The index of the current chain of auxiliary variables. Defaults to 1.

        Returns:
            The transformed expression.
        """
        assert isinstance(expression, (sp.Add, sp.Mul)), (
            f"We expect a sum of products or a single product as input but got {expression}"
        )
        problematic_terms = (
            self.__get_problematic_terms(cast("Collection[sp.Mul]", expression.args))
            if isinstance(expression, sp.Add)
            else self.__get_problematic_terms([expression])
        )

        if not problematic_terms:
            return expression

        counts: dict[tuple[sp.Expr, sp.Expr], int] = {}
        highest_key: tuple[sp.Expr, sp.Expr] | None = None
        for term in problematic_terms:
            symbols = sorted(term.free_symbols, key=str, reverse=True)
            for s1 in symbols:
                assert isinstance(s1, sp.Expr)
                for s2 in symbols:
                    assert isinstance(s2, sp.Expr)
                    if s1 == s2:
                        continue
                    key = (s1, s2)
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += 1
                    if highest_key is None or counts[key] > counts[highest_key]:
                        highest_key = key
        if highest_key is None:
            msg = "Error reducing cost function order: No highest key found, but problematic terms still exist."
            raise RuntimeError(msg)

        selected_variables = [highest_key[0], highest_key[1]]
        equality_penalty: sp.Expr = sp.Integer(0)
        if any(str(selected_variables[0]) in x for x in assignment.slack_dict.values()):
            old_var = selected_variables[0]
            current = selected_variables[0]
            while any(str(current) in x for x in assignment.slack_dict.values()):
                current = sp.Symbol(f"{current}'")
            selected_variables[0] = current
            equality_penalty += (
                old_var + selected_variables[0] - old_var * selected_variables[0]
            )  # introduce (x - x')^2 as penalty.
        if any(str(selected_variables[1]) in x for x in assignment.slack_dict.values()):
            old_var = selected_variables[1]
            current = selected_variables[1]
            while any(str(current) in x for x in assignment.slack_dict.values()):
                current = sp.Symbol(f"{current}'")
            selected_variables[1] = current
            equality_penalty += (
                old_var + selected_variables[1] - 2 * old_var * selected_variables[1]
            )  # introduce (x - x')^2 as penalty.

        y = sp.Symbol(f"y_{chain_index}_1")
        assignment.add_slack_variable(str(y), str(selected_variables[0]), str(selected_variables[1]))
        new_expr = expression.subs({highest_key[0] * highest_key[1]: y}) + QuboGenerator.__get_slack_penalty(
            selected_variables[0], selected_variables[1], y
        )
        new_expr += equality_penalty
        return self.__continue_slack_chain(new_expr, y, assignment, chain_index, 1)

    @classmethod
    def __get_slack_penalty(cls, x1: sp.Expr, x2: sp.Expr, y: sp.Symbol) -> sp.Expr:
        """Computes the slack penalty for a given pair of variables and an auxiliary variable.

        Args:
            x1: The first variable.
            x2: The second variable.
            y: The auxiliary variable.

        Returns:
            The slack penalty expression.
        """
        return 100 * x1 * x2 - 200 * y * x1 - 200 * y * x2 + 300 * y

    def __get_problematic_terms(self, terms: Collection[sp.Mul]) -> list[sp.Mul]:
        problematic_terms: list[sp.Mul] = []
        for term in terms:
            unpowered = self.__unpower(term)
            order = self.__get_order(unpowered)
            if order <= 2:
                continue
            assert isinstance(unpowered, sp.Mul), "Only products of variables can be left from sums of products."
            problematic_terms.append(unpowered)
        return problematic_terms

    def __continue_slack_chain(
        self,
        expression: sp.Expr,
        last_slack: sp.Symbol,
        assignment: SlackChainAssignment,
        chain_index: int,
        index: int,
    ) -> sp.Expr:
        problematic_terms = self.__get_problematic_terms(cast("Collection[sp.Mul]", expression.args))

        if not problematic_terms:
            return expression

        counts: dict[sp.Expr, int] = {}
        highest_key: sp.Expr | None = None
        for term in problematic_terms:
            symbols = sorted(term.free_symbols, key=str, reverse=True)
            if last_slack not in symbols:
                continue
            for s in symbols:
                assert isinstance(s, sp.Expr)
                if s == last_slack:
                    continue
                if s not in counts:
                    counts[s] = 0
                counts[s] += 1
                if highest_key is None or counts[s] > counts[highest_key]:
                    highest_key = s

        if highest_key is None:
            return self.expand_higher_order_terms(expression, assignment, chain_index + 1)
        selected_replacement = highest_key
        equality_penalty: sp.Expr = sp.Integer(0)
        if any(str(selected_replacement) in x for x in assignment.slack_dict.values()):
            while any(str(selected_replacement) in x for x in assignment.slack_dict.values()):
                selected_replacement = sp.Symbol(f"{selected_replacement}'")
            equality_penalty += (
                100 * highest_key + 100 * selected_replacement - 200 * highest_key * selected_replacement
            )  # introduce (x - x')^2 as penalty.
        y = sp.Symbol(f"y_{chain_index}_{index + 1}")
        assignment.add_slack_variable(str(y), str(last_slack), str(selected_replacement))
        new_expr = expression.subs({last_slack * highest_key: y}) + self.__get_slack_penalty(
            last_slack, selected_replacement, y
        )
        new_expr += equality_penalty
        return self.__continue_slack_chain(new_expr, y, assignment, chain_index, index + 1)

    def expand_higher_order_terms_greedy_minimization(
        self, expression: sp.Expr, assignment: SlackChainAssignment
    ) -> sp.Expr:
        """Reduce the order of higher order terms in a cost function by employing greedy variable minimization.

        Generally, attempts to keep the number of auxiliary variables low by selecting the most frequently occurring
        products of variables to replace first.

        Args:
            expression: The expression to transform.
            assignment: The assignment to store information about the slack variables.

        Returns:
            The transformed expression.
        """
        assert isinstance(expression, (sp.Add, sp.Mul)), (
            f"We expect a sum of products or a single product as input but got {expression}"
        )
        slack_count = 0

        while True:
            problematic_terms = (
                self.__get_problematic_terms(cast("Collection[sp.Mul]", expression.args))
                if isinstance(expression, sp.Add)
                else self.__get_problematic_terms([expression])
                if isinstance(expression, sp.Mul)
                else None
            )
            if problematic_terms is None:
                msg = f"Error reducing cost function order: Expected a sum of products or a single product but got {expression}"
                raise TypeError(msg)

            if not problematic_terms:
                return expression

            counts: dict[tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function], int] = {}
            highest_key: tuple[sp.Symbol | sp.Function, sp.Symbol | sp.Function] | None = None
            for term in problematic_terms:
                symbols = sorted(
                    [symbol for symbol in term.args if isinstance(symbol, (sp.Function, sp.Symbol))], key=str
                )
                for s1 in symbols:
                    for s2 in symbols:
                        if s1 == s2:
                            continue
                        key = (s1, s2)
                        if key not in counts:
                            counts[key] = 0
                        counts[key] += 1
                        if highest_key is None or counts[key] > counts[highest_key]:
                            highest_key = key

            if highest_key is None:
                msg = "Error reducing cost function order: No highest key found, but problematic terms still exist."
                raise RuntimeError(msg)
            selected_variables = [highest_key[0], highest_key[1]]

            slack_count += 1
            y = sp.Symbol(f"y_{slack_count}")
            assignment.add_slack_variable(str(y), str(selected_variables[0]), str(selected_variables[1]))
            expression = expression.subs({highest_key[0] * highest_key[1]: y}) + self.__get_slack_penalty(
                selected_variables[0], selected_variables[1], y
            )

    def __get_order(self, expression: sp.Expr) -> int:
        """Computes the order of a product of variables.

        Args:
            expression: The expression to check.

        Returns:
            The order of the expression. If the expression is not a product, 1 is returned.
        """
        if isinstance(expression, sp.Mul):
            return sum(self.__get_order(arg) for arg in expression.args)
        if isinstance(expression, (sp.Symbol, sp.Function)):
            return 1
        return 0

    def __unpower(self, expression: sp.Expr) -> sp.Expr:
        """Removes exponentiation from an expression.

        This matters, because when using binary variables, x^2 = x always holds. This allows us to compute the order of the
        term more easily.

        Args:
            expression: The expression to transform.

        Returns:
            The transformed expression.
        """
        if isinstance(expression, sp.Pow):
            return expression.args[0]
        if isinstance(expression, sp.Mul):
            return sp.Mul(*[self.__unpower(arg) for arg in expression.args])
        return expression

    def __get_auxiliary_variables(self, expression: sp.Expr) -> list[sp.Symbol]:
        """Returns a list of all auxiliary variables used in an expression.

        Auxiliary variables will start with "y_" by definition.

        Args:
            expression: The expression to check.

        Returns:
            The list of employed auxiliary variables.
        """
        if isinstance(expression, sp.Mul):
            return list({var for arg in expression.args for var in self.__get_auxiliary_variables(arg)})
        if isinstance(expression, sp.Symbol) and (str(expression).startswith("y_") or str(expression).endswith("'")):
            return [expression]
        return []

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        """A method that can be extended by classes that inherit from QuboGenerator to transform the QUBO formulation into expanded form, if that process requires additional steps.

        Args:
            expression: The expression to transform.

        Returns:
            The transformed expression.
        """
        return expression

    def construct_qubo_matrix(self, for_embedding: bool = False) -> npt.NDArray[np.int_ | np.float64]:
        """Constructs the matrix representation of the QUBO problem.

        This is achieved by first creating the expanded QUBO formula, and then taking the coefficients of each term.

        Returns:
            The matrix representation of the QUBO problem.
        """
        if not for_embedding:
            expansion = self.construct_expansion(for_embedding=for_embedding)
            expansion = cast("sp.Expr", expansion)
            coefficients = dict(expansion.expand().as_coefficients_dict())
            auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
            auxiliary_variables.sort(
                key=lambda var: tuple(
                    [0 if str(var).startswith("y") else 1] + [int(x) for x in str(var)[2:].replace("'", "").split("_")]
                )
            )
            result = np.zeros((
                self.get_encoding_variable_count() + len(auxiliary_variables),
                self.get_encoding_variable_count() + len(auxiliary_variables),
            ))

            all_variables = dict(self._get_encoding_variables())

            def get_index(variable: sp.Expr) -> int:
                if variable in all_variables:
                    return all_variables[variable] - 1
                return auxiliary_variables.index(cast("sp.Symbol", variable)) + self.get_encoding_variable_count()

        else:
            expansion = self.construct_expansion(include_slack_information=True, for_embedding=for_embedding)
            expansion = cast("tuple[sp.Expr, SlackChainAssignment]", expansion)
            expression, assignment = expansion
            num_variables = len(assignment.indices)
            result = np.zeros((
                num_variables,
                num_variables,
            ))
            coefficients = dict(expression.as_coefficients_dict())

            def get_index(variable: sp.Expr) -> int:
                return assignment.indices[str(variable)]

        for term, value in coefficients.items():
            if isinstance(term, sp.Mul):
                index1 = get_index(term.args[0])
                index2 = get_index(term.args[1])
                if index1 > index2:
                    index1, index2 = index2, index1
                result[index1][index2] = value
            elif isinstance(term, (sp.Symbol, sp.Function)):
                index = get_index(term)
                result[index][index] = value

        return result

    def get_cost(self, assignment: list[int], for_embedding: bool = False) -> float:
        """Given an assignment, computes the total cost value of the corresponding cost function evaluation.

        The assignment is given as a binary list, and can either contain assignments for just encoding variables
        or encoding + auxiliary variables. In the former case, the auxiliary values are computed automatically.

        Args:
            assignment: The assignment for each variable (either 0 or 1).
            for_embedding: Whether to prepare the QUBO for embedding on a quantum device. Defaults to False.

        Returns:
            The cost value for the assignment.
        """
        if any(x not in {0, 1} for x in assignment):
            msg = "Provided values are not binary (1/0)"
            raise ValueError(msg)

        full_expansion = self.construct_expansion(include_slack_information=True, for_embedding=for_embedding)
        full_expansion = cast("tuple[sp.Expr, SlackChainAssignment]", full_expansion)
        expansion, auxiliary_expansions = full_expansion

        auxiliary_assignment: dict[sp.Expr, int] = {}
        if len(assignment) == self.get_encoding_variable_count():
            auxiliary_assignment = self.__get_auxiliary_assignment(assignment, auxiliary_expansions)
        elif len(assignment) != self.count_required_variables(for_embedding=for_embedding):
            msg = "Invalid assignment length."
            raise ValueError(msg)

        variable_assignment = {item[0]: assignment[item[1] - 1] for item in self._get_encoding_variables()}
        variable_assignment.update(auxiliary_assignment)
        return expansion.subs(variable_assignment).evalf()  # ty: ignore[no-matching-overload]

    def solve_brute_force(self) -> tuple[list[int], float]:
        """Finds an optimal assignment for the QUBO problem by evaluating every possible assignment.

        The search covers all variables of the QUBO matrix, including auxiliary variables. The runtime grows as
        2^n for n variables, so this method is only practical for small problems.

        Returns:
            An optimal assignment in the variable order of the QUBO matrix, and its cost, including constant terms.
        """
        qubo = self.construct_qubo_matrix()
        num_variables = len(qubo)
        expansion = cast("sp.Expr", self.construct_expansion())
        constant = float(dict(expansion.as_coefficients_dict()).get(sp.Integer(1), 0))

        num_states = 1 << num_variables
        batch_size = 1 << min(num_variables, 16)
        positions = np.arange(num_variables)
        best_state = 0
        best_cost = np.inf
        for start in range(0, num_states, batch_size):
            states = np.arange(start, min(start + batch_size, num_states))
            bits = (states[:, np.newaxis] >> positions) & 1
            costs = np.sum((bits @ qubo) * bits, axis=1)
            best_index = int(np.argmin(costs))
            if costs[best_index] < best_cost:
                best_cost = float(costs[best_index])
                best_state = int(states[best_index])

        return [(best_state >> i) & 1 for i in range(num_variables)], best_cost + constant

    def __get_auxiliary_assignment(
        self, assignment: list[int], auxiliary_expansions: SlackChainAssignment
    ) -> dict[sp.Expr, int]:
        """Generates the assignment of auxiliary variables based on a given encoding variable assignment.

        Every auxiliary variable is defined like `y_k = x_i * x_j`, `y_k = x_i * y_j` or `y_k = y_i * y_j`.
        These definitions are stored in the `auxiliary_cache` dictionary. As there are no circular references,
        a given set of assignments for each `x_i` is enough to compute the values of all `y_k`. This can be used to
        compute the cost for a given QUBO encoding without requiring the user to define all auxiliary variables.

        Args:
            assignment: The assignment for the encoding variables `x_i`.
            auxiliary_expansions: Information about the auxiliary variable expansions.

        Returns:
            An assignment dictionary, mapping each `y_k` to its value in {0, 1}.
        """
        auxiliary_values: dict[str, int] = {}
        encoding_variables = {str(key): value for key, value in self._get_encoding_variables()}
        remaining_variables = set(auxiliary_expansions.slack_dict.keys())

        def get_var_value(var: str) -> int | None:
            if var in encoding_variables:
                return assignment[encoding_variables[var] - 1]
            if var in auxiliary_values:
                return auxiliary_values[var]
            return None

        while remaining_variables:
            to_remove = []
            for aux in remaining_variables:
                (left, right) = auxiliary_expansions.slack_dict[aux]
                left_val = get_var_value(left)
                right_val = get_var_value(right)
                if left_val is not None and right_val is not None:
                    auxiliary_values[aux] = left_val * right_val
                    to_remove.append(aux)
            for aux in to_remove:
                remaining_variables.remove(aux)

        return {sp.Symbol(key): value for key, value in auxiliary_values.items()}

    def _get_encoding_variables(self) -> list[tuple[sp.Expr, int]]:
        """Returns all non-auxiliary variables used in the QUBO formulation.

        Returns:
            A list of tuples containing the variable and its index.
        """
        all_expressions: list[sp.Expr] = [self.objective_function] if self.objective_function is not None else []
        all_expressions += [penalty[0] for penalty in self.penalties]
        variables = set()
        for expr in all_expressions:
            variables |= expr.free_symbols
        sorted_labels = sorted(variables, key=lambda var: int(str(var)[2:]))
        return [(var, i + 1) for i, var in enumerate(sorted_labels)]

    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        """Computes the penalty factors for each constraint. May be extended by subclasses.

        Returns:
            A list of tuples containing the individual cost functions and their constraints.
        """
        return [(expr, weight) if weight is not None else (expr, 1.0) for (expr, weight) in self.penalties]

    def count_required_variables(self, for_embedding: bool = False) -> int:
        """Returns the total number of variables required to represent the QUBO problem.

        Args:
            for_embedding: Whether to prepare the QUBO for embedding on a quantum device. Defaults to False.

        Returns:
            The number of required variables.
        """
        expansion = self.construct_expansion(for_embedding=for_embedding)
        expansion = cast("sp.Expr", expansion)
        coefficients = dict(expansion.as_coefficients_dict())
        auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
        return len(self._get_encoding_variables()) + len(auxiliary_variables)

    def get_encoding_variable_count(self) -> int:
        """Returns the number of non-auxiliary binary variables required to represent the QUBO problem.

        Returns:
            The number of required binary variables.
        """
        return len(self._get_encoding_variables())

    def get_variable_index(self, variable: sp.Expr) -> int:
        """For a given variable, returns its index in the QUBO matrix.

        Args:
            variable: The variable to investigate.

        Returns:
            The index of the variable.
        """
        encoding_variables = [x[0] for x in self._get_encoding_variables()]
        if variable in encoding_variables:
            return encoding_variables.index(variable) + 1
        return -1

    def decode_bit_array(self, _array: list[int]) -> Any:  # ruff:ignore[any-type]
        """Given an assignment, decodes it into a meaningful result. May be extended by subclasses.

        Args:
            _array: The binary assignment.

        Returns:
            The decoded result.
        """
        return ""

    def construct_qaoa_circuit(
        self, n_qubits: int = -1, do_reuse: bool = False, include_barriers: bool = False
    ) -> qiskit.QuantumCircuit:
        """Constructs a QAOA circuit for the QUBO problem.

        The cost layer applies the phase exp(-i 4 gamma f(x)) to each basis state x, up to a global phase.
        Each `rzz` gate encodes a quadratic QUBO coefficient, and each `rz` gate encodes the linear field of a variable.
        The mixer layer applies exp(-i beta X) to each qubit.

        Args:
            n_qubits: If given, sets the number of qubits for the full circuit.
            do_reuse: Attempt to reuse qubits to limit the number of variables. Defaults to False.
            include_barriers: Whether to include barriers in the circuit. Defaults to False.

        Returns:
            The constructed QAOA circuit.
        """
        qubo = self.construct_qubo_matrix(for_embedding=False)
        qubits = len(qubo)
        interactions = [(i, j) for i in range(qubits) for j in range(i + 1, qubits) if qubo[i][j] != 0]
        # Substituting x_i = (1 - z_i) / 2 turns each Q_ij x_i x_j into Ising terms on z_i z_j, z_i, and z_j.
        linear_fields = [
            2 * float(qubo[i][i]) + sum(float(qubo[i][j] + qubo[j][i]) for j in range(qubits) if j != i)
            for i in range(qubits)
        ]
        circuit = qiskit.QuantumCircuit(n_qubits if n_qubits != -1 else qubits, qubits)
        gamma = qiskit.circuit.Parameter("gamma")
        beta = qiskit.circuit.Parameter("beta")

        def add_rzz(i: int, j: int, qubit_i: int, qubit_j: int) -> None:
            circuit.rzz(2 * gamma * float(qubo[min(i, j)][max(i, j)]), qubit_i, qubit_j)

        def add_rz(i: int, qubit: int) -> None:
            if linear_fields[i] != 0:
                circuit.rz(-2 * gamma * linear_fields[i], qubit)

        if not do_reuse:
            for i in range(qubits):
                circuit.h(i)
            if include_barriers:
                circuit.barrier()

            for i, j in interactions:
                add_rzz(i, j, i, j)
            for i in range(qubits):
                add_rz(i, i)
            if include_barriers:
                circuit.barrier()

            for i in range(qubits):
                circuit.rx(2 * beta, i)
            if include_barriers:
                circuit.barrier()
            for i in range(qubits):
                circuit.measure(i, i)
        else:
            used_qubits = set()
            free_qubits = list(range(qubits))
            substitution: dict[int, int] = {}
            outgoing: dict[int, set[int]] = {i: set() for i in range(qubits)}
            for i, j in interactions:
                outgoing[i].add(j)
                outgoing[j].add(i)

            def add_new_qubit(qubit: int) -> None:
                substitution[qubit] = free_qubits.pop(0)
                if substitution[qubit] in used_qubits:
                    circuit.reset(substitution[qubit])
                else:
                    used_qubits.add(substitution[qubit])
                circuit.h(substitution[qubit])

            total_outgoing: set[int] = set()
            remaining = list(range(qubits))
            covered = set()
            while remaining:
                best_size = qubits + 1
                next_qubit = -1
                for q in remaining:
                    size = len(total_outgoing.union(outgoing[q]))
                    if size < best_size or (q in remaining and next_qubit not in remaining):
                        best_size = size
                        next_qubit = q
                remaining.remove(next_qubit)
                total_outgoing = total_outgoing.union(outgoing[next_qubit])
                if next_qubit not in substitution:
                    add_new_qubit(next_qubit)

                for other in outgoing[next_qubit]:
                    if other in covered:
                        continue
                    if other not in substitution:
                        add_new_qubit(other)
                    add_rzz(next_qubit, other, substitution[next_qubit], substitution[other])
                add_rz(next_qubit, substitution[next_qubit])
                covered.add(next_qubit)
                circuit.rx(2 * beta, substitution[next_qubit])
                circuit.measure(substitution[next_qubit], next_qubit)
                free_qubits.insert(0, substitution[next_qubit])

        if do_reuse and len(used_qubits) < len(circuit.qubits):
            return self.construct_qaoa_circuit(n_qubits=len(used_qubits), do_reuse=True)
        return circuit

    def construct_interaction_graph(self, offset: int = 0, for_embedding: bool = False) -> list[tuple[int, int]]:
        """Constructs the interaction graph of the QUBO problem.

        Returns:
            The interaction graph of the QUBO problem.
        """
        qubo = self.construct_qubo_matrix(for_embedding=for_embedding)
        return [(i + offset, j + offset) for i in range(len(qubo)) for j in range(i + 1, len(qubo)) if qubo[i][j] != 0]

    def construct_embedded_qaoa_circuit(self, device: Calibration) -> qiskit.QuantumCircuit:
        """Constructs a QAOA circuit for the QUBO problem that is embedded on the given device.

        Args:
            device: The device to embed the QAOA circuit on.

        Returns:
            The constructed QAOA circuit.
        """
        expansion = self.construct_expansion(include_slack_information=True, for_embedding=True)
        expansion = cast("tuple[sp.Expr, SlackChainAssignment]", expansion)
        expression, assignment = expansion
        coefficients = dict(expression.as_coefficients_dict())
        auxiliary_variables = list({var for arg in coefficients for var in self.__get_auxiliary_variables(arg)})
        auxiliary_variables.sort(
            key=lambda var: tuple(
                [0 if str(var).startswith("y") else 1] + [int(x) for x in str(var)[2:].replace("'", "").split("_")]
            )
        )

        variable_indices = assignment.indices
        slack_dict = assignment.slack_dict
        indices_to_variables = {i: v for v, i in variable_indices.items()}

        chains = assignment.chains
        interactions = self.construct_interaction_graph(offset=0, for_embedding=True)
        covered_interactions: set[tuple[int, int]] = set()

        substitution: dict[str, int] = {}
        full_nn_chain: list[int] = device.get_connected_qubit_chain()
        nn_qubit_queue: list[int] = device.get_connected_qubit_chain()

        qc = qiskit.QuantumCircuit(device.num_qubits, device.num_qubits)
        beta = qiskit.circuit.Parameter("beta")
        gamma = qiskit.circuit.Parameter("gamma")

        def add_rzz(q1: str, q2: str) -> None:
            nonlocal qc, gamma, substitution
            angle = float(coefficients[sp.Symbol(q1) * sp.Symbol(q2)])
            qc.rzz(gamma * 2 * angle, substitution[q1], substitution[q2])

        def add_rz(q: str, angle: float) -> None:
            nonlocal qc, gamma, substitution
            qc.rz(gamma * 2 * angle, substitution[q])

        def add_swap(q1: str, q2: str) -> None:
            nonlocal qc, gamma, substitution
            qc.swap(substitution[q1], substitution[q2])

        def get_next_substitution(q: str) -> int:
            nonlocal substitution, nn_qubit_queue
            if q in substitution:
                return substitution[q]
            substitution[q] = nn_qubit_queue.pop(0)
            return substitution[q]

        chain_groups: list[list[tuple[str, str]]] = []
        for chain in chains:
            first_var = chain[0]
            first_pair = slack_dict[first_var]
            get_next_substitution(first_pair[0])
            get_next_substitution(first_pair[1])
            groups: list[
                tuple[str, str]
            ] = []  # A group always consists of a slack var and its subsequent encoding var, or the first two encoding variables
            groups.append(first_pair)
            for slack in chain[1:]:
                predecessors = slack_dict[slack]
                last_slack = predecessors[0]
                shared_var = predecessors[1]
                get_next_substitution(last_slack)
                get_next_substitution(shared_var)
                groups.append((last_slack, shared_var))
            substitution[chain[-1]] = nn_qubit_queue.pop(0)
            groups.append((chain[-1], ""))
            chain_groups.append(groups)

        for groups in chain_groups:
            for group in groups:
                if not group[1]:
                    continue
                add_rzz(group[0], group[1])
                covered_interactions.add((variable_indices[group[0]], variable_indices[group[1]]))
        for groups in chain_groups:
            for g1, g2 in pairwise(groups):
                add_rzz(g1[1], g2[0])
                covered_interactions.add((variable_indices[g1[1]], variable_indices[g2[0]]))
                covered_interactions.add((
                    variable_indices[g1[0]],
                    variable_indices[g2[0]],
                ))  # will be added in for loop below but needs to be done now already for next step

        for i, j in interactions:
            if (i, j) in covered_interactions or (j, i) in covered_interactions:
                continue
            v_i = indices_to_variables[i]
            v_j = indices_to_variables[j]
            if v_i not in substitution:
                substitution[v_i] = nn_qubit_queue.pop(0)
            if v_j not in substitution:
                substitution[v_j] = nn_qubit_queue.pop(0)
            add_rzz(v_i, v_j)

        incomplete_swaps = {}
        for groups in chain_groups:
            for g1, g2 in zip(groups[::2], groups[1::2], strict=False):
                add_swap(g1[0], g1[1])
                add_rzz(g1[1], g2[0])
                add_swap(g1[0], g1[1])
            for g1, g2 in zip(groups[1::2], groups[2::2], strict=False):
                incomplete_swaps[g1[0]] = g1[1]
                incomplete_swaps[g1[1]] = g1[0]
                add_swap(g1[0], g1[1])
                add_rzz(g1[1], g2[0])

        for q in expression.free_symbols:
            assert isinstance(q, sp.Expr)
            total_angle = coefficients.get(q, 0) * 2
            for q_2 in expression.free_symbols:
                assert isinstance(q_2, sp.Expr)
                if q_2 == q:
                    continue
                total_angle += coefficients.get(q * q_2, 0)
            if total_angle != 0:
                add_rz(str(q), -float(total_angle))
        qc_front = qiskit.QuantumCircuit(device.num_qubits, device.num_qubits)
        for i in range(len(full_nn_chain) - len(nn_qubit_queue)):
            q = full_nn_chain[i]

            qc_front.h(q)
            qc.rx(2 * beta, q)
        for name, idx in substitution.items():
            measure_index = substitution[incomplete_swaps[name]] if name in incomplete_swaps else idx
            qc.measure(measure_index, assignment.indices[name])
        return qc_front.compose(qc, front=False, inplace=False)
