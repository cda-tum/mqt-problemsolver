# Import of the needed libraries

from __future__ import annotations

import math

import numpy as np
from qubovert import PUBO
from src.mqt.qao import Constraints, ObjectiveFunction, Variables


class Problem:
    """class of the problem to solve
    It contains object variables,
    an object constraints, an object
    cost function and an object solver.
    It creates the connection among the
    various methods for automatizing the procedure"""

    def __init__(self) -> None:
        """declaration of the needed object"""
        self.variables: Variables = Variables()
        self.auxiliary_variables: Variables = Variables()
        self.constraints: Constraints = Constraints()
        self.objective_function: ObjectiveFunction = ObjectiveFunction()
        self.pubo: PUBO = PUBO()
        self.pubo_cost_function_alone: PUBO = PUBO()
        self.lambdas: list[float] = []
        self.penalty_weights_upper_bound: dict[int, float] = {}

    def create_problem(self, var: Variables, constraint: Constraints, objective_functions: ObjectiveFunction) -> None:
        """function for creating to define the problem characteristic

        Keyword arguments:
        var -- Variables involved in the problem
        constraint -- constraints to take into account in the problem
        objective_function -- function to optimize


        Return values:
        None

        """
        self.variables = var
        self.constraints = constraint
        self.objective_function = objective_functions

    def write_the_final_cost_function(self, lambda_strategy: str, lambda_value: float = 1) -> PUBO:
        """function for writing the final PUBO function

        Keyword arguments:
        lambda strategy to choose for putting together the penalty functions and the cost function


        Return values:
        final pubo function

        """
        self.variables.move_to_binary(self.constraints.constraints)
        self.constraints.translate_constraints(self.variables)
        self.pubo = self.objective_function.rewrite_cost_functions(self.pubo, self.variables)
        self.pubo_cost_function_alone = self.pubo
        if lambda_strategy == "upper_bound_only_positive":
            lambda_val = self._upper_bound_with_only_positive_coefficient(self.pubo)
            if isinstance(lambda_val, float):
                self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
            else:
                lambda_val = self.upper_lower_bound_naive_method(self.pubo)
                self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        elif lambda_strategy == "maximum_coefficient":
            lambda_val = self._maximum_qubo_coefficient(self.pubo)
            self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        elif lambda_strategy == "VLM":
            lambda_val = self._vlm(self.pubo)
            self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        elif lambda_strategy == "MOMC":
            for elm in self.constraints.constraints_penalty_functions:
                self.lambdas.append(self._momc(self.pubo, elm[0]))
        elif lambda_strategy == "MOC":
            for elm in self.constraints.constraints_penalty_functions:
                self.lambdas.append(self._moc(self.pubo, elm[0]))
        elif lambda_strategy == "upper lower bound naive":
            lambda_val = self.upper_lower_bound_naive_method(self.pubo)
            self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        elif lambda_strategy == "upper lower bound posiform and negaform method":
            lambda_val = self.upper_lower_bound_posiform_and_negaform_method(self.pubo)
            self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        elif lambda_strategy == "manual":
            self.lambdas = [lambda_value] * len(self.constraints.constraints_penalty_functions)
        else:
            lambda_val = self.upper_lower_bound_naive_method(self.pubo)
            self.lambdas = [lambda_val] * len(self.constraints.constraints_penalty_functions)
        for j in range(len(self.constraints.constraints_penalty_functions)):
            el = self.constraints.constraints_penalty_functions[j]
            if el[1]:  # Strong constraint
                self.pubo += 1.1 * self.lambdas[j] * el[0]
            else:  # Weak constraint
                self.pubo += 0.9 * self.lambdas[j] * el[0]
        return self.pubo

    def update_lambda_cost_function(
        self,
        single_satisfied: list[bool],
        maximum_number_of_update: int = 1,
        update_strategy: str = "sequential penalty increase",
    ) -> PUBO:
        """function for updating the penalty weights

        Keyword arguments:
        single_satisfied -- list of bool containing the information about which are the not satisfied constraint
        maximum_number_of_update -- int contain the maximum number of admitted lambda updates  (needed for same update strategy)
        update_strategy -- str for identifying the wanted update strategy

        Return values:
        final pubo function

        """
        self.pubo = self.pubo_cost_function_alone.copy()
        for j in range(len(self.constraints.constraints_penalty_functions)):
            el = self.constraints.constraints_penalty_functions[j]
            if single_satisfied[j]:
                if el[1]:  # Strong constraint
                    self.pubo += 1.1 * self.lambdas[j] * el[0]
                else:  # Weak constraint
                    self.pubo += 0.9 * self.lambdas[j] * el[0]
            else:
                if update_strategy == "sequential penalty increase":
                    self.lambdas[j] = self._sequential_penalty_increase(self.lambdas[j])
                elif update_strategy == "scaled sequential penalty increase":
                    if j not in self.penalty_weights_upper_bound:
                        self.penalty_weights_upper_bound[j] = self.lambdas[j] * 100
                    self.lambdas[j] = self._scaled_sequential_penalty_increase(
                        self.lambdas[j], self.penalty_weights_upper_bound[j], maximum_number_of_update
                    )

                elif update_strategy == "binary search penalty algorithm":
                    if j not in self.penalty_weights_upper_bound:
                        self.penalty_weights_upper_bound[j] = self.lambdas[j] * 100
                    self.lambdas[j] = self._binary_search_penalty_algorithm(
                        self.lambdas[j], self.penalty_weights_upper_bound[j]
                    )
                else:
                    self.lambdas[j] = self._sequential_penalty_increase(self.lambdas[j])
                if el[1]:  # Strong constraint
                    self.pubo += 1.1 * self.lambdas[j] * el[0]
                else:  # Weak constraint
                    self.pubo += 0.9 * self.lambdas[j] * el[0]

        return self.pubo

    def _upper_bound_with_only_positive_coefficient(self, cost_function: PUBO) -> float | bool:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        or False if the coefficients are not all positive

        """
        upperbound = 0.0
        for key in cost_function:
            if len(key) > 0:
                if cost_function[key] > 0:
                    upperbound += cost_function[key]
                elif cost_function[key] < 0:
                    return False
            else:
                upperbound += cost_function[key]
        return upperbound

    def _maximum_qubo_coefficient(self, cost_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        max_coeff = 0.0
        offset = 0.0
        first = True
        for key in cost_function:
            if len(key) > 0:
                if first:
                    max_coeff = cost_function[key]
                    first = False
                elif cost_function[key] > max_coeff:
                    max_coeff = cost_function[key]
            else:
                offset = cost_function[key]
        return max_coeff + offset

    def _vlm(self, cost_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        p_sum = {}
        n_sum = {}
        for var in cost_function.variables:
            p_sum[var] = 0
            n_sum[var] = 0
        for key in cost_function:
            if len(key) == 1:
                p_sum[key[0]] += cost_function[key]
                n_sum[key[0]] -= cost_function[key]
            elif len(key) != 0:
                if cost_function[key] > 0:
                    for elem in key:
                        p_sum[elem] += cost_function[key]
                elif cost_function[key] < 0:
                    for elem in key:
                        n_sum[elem] -= cost_function[key]
        return float(np.max([np.array(list(p_sum.values())), np.array(list(n_sum.values()))]))

    def _momc(self, cost_function: PUBO, constraint_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        p_sum = {}
        n_sum = {}
        for var in cost_function.variables:
            p_sum[var] = 0
            n_sum[var] = 0
        for key in cost_function:
            if len(key) == 1:
                p_sum[key[0]] += cost_function[key]
                n_sum[key[0]] -= cost_function[key]
            elif len(key) != 0:
                if cost_function[key] > 0:
                    for elem in key:
                        p_sum[elem] += cost_function[key]
                elif cost_function[key] < 0:
                    for elem in key:
                        n_sum[elem] -= cost_function[key]
        wc_max = float(np.max([np.array(list(p_sum.values())), np.array(list(n_sum.values()))]))

        p_sum_constrained = {}
        n_sum_constrained = {}
        for var in constraint_function.variables:
            p_sum_constrained[var] = 0
            n_sum_constrained[var] = 0
        for key in constraint_function:
            if len(key) == 1:
                p_sum_constrained[key[0]] += constraint_function[key]
                n_sum_constrained[key[0]] -= constraint_function[key]
            elif len(key) != 0:
                if constraint_function[key] > 0:
                    for elem in key:
                        p_sum_constrained[elem] += constraint_function[key]
                elif constraint_function[key] < 0:
                    for elem in key:
                        n_sum_constrained[elem] -= constraint_function[key]
        wg_min = min(list(p_sum_constrained.values()) + list(n_sum_constrained.values()))
        if wg_min == 0:
            return wc_max
        return max(1.0, wc_max / wg_min)

    def _moc(self, cost_function: PUBO, constraint_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        p_sum = {}
        n_sum = {}
        for var in cost_function.variables:
            p_sum[var] = 0
            n_sum[var] = 0
        for key in cost_function:
            if len(key) == 1:
                p_sum[key[0]] += cost_function[key]
                n_sum[key[0]] -= cost_function[key]
            elif len(key) != 0:
                if cost_function[key] > 0:
                    for elem in key:
                        p_sum[elem] += cost_function[key]
                elif cost_function[key] < 0:
                    for elem in key:
                        n_sum[elem] += cost_function[key]

        p_sum_constrained = {}
        n_sum_constrained = {}
        for var in constraint_function.variables:
            p_sum_constrained[var] = 0
            n_sum_constrained[var] = 0
        for key in constraint_function:
            if len(key) == 1:
                p_sum_constrained[key[0]] += constraint_function[key]
                n_sum_constrained[key[0]] -= constraint_function[key]
            elif len(key) != 0:
                if constraint_function[key] > 0:
                    for elem in key:
                        p_sum_constrained[elem] += constraint_function[key]
                elif constraint_function[key] < 0:
                    for elem in key:
                        n_sum_constrained[elem] += constraint_function[key]

        val = 0.0
        first = True
        for key in p_sum:
            if key in p_sum_constrained and p_sum_constrained[key] != 0:
                v = p_sum[key] / p_sum_constrained[key]
                if v != 0 and first:
                    val = v
                    first = False
                elif v > val:
                    val = v

            if key in n_sum_constrained and n_sum_constrained[key] != 0:
                v = n_sum[key] / n_sum_constrained[key]
                if v != 0 and first:
                    val = v
                    first = False
                elif v > val:
                    val = v
        return max(1, val)

    def upper_lower_bound_naive_method(self, cost_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        upper_bound = 0.0
        lower_bound = 0.0
        for key in cost_function:
            if len(key) >= 1:
                if cost_function[key] > 0:
                    upper_bound += cost_function[key]
                elif cost_function[key] < 0:
                    lower_bound += cost_function[key]
        return upper_bound - lower_bound

    def upper_lower_bound_posiform_and_negaform_method(self, cost_function: PUBO) -> float:
        """function for estimating the weights for constraints

        Keyword arguments:
        cost function -- is the model of the main problem cost function


        Return values:
        lambda -- return the weight for the constraints
        """
        p_sum = {}
        n_sum = {}
        for var in cost_function.variables:
            p_sum[var] = 0
            n_sum[var] = 0
        for key in cost_function:
            if len(key) == 1:
                p_sum[key[0]] += cost_function[key]
                n_sum[key[0]] += cost_function[key]
            elif len(key) > 1:
                if cost_function[key] < 0:
                    for elem in key:
                        p_sum[elem] += cost_function[key]
                elif cost_function[key] > 0:
                    for elem in key:
                        n_sum[elem] += cost_function[key]
        lowerbound = 0.0
        upperbound = 0.0
        for key in p_sum:
            if p_sum[key] < 0:
                lowerbound += p_sum[key]
            if n_sum[key] > 0:
                upperbound += n_sum[key]
        return upperbound - lowerbound

    def _sequential_penalty_increase(self, current_lambda: float) -> float:
        """function for updating weights for constraints

        Keyword arguments:
        current_lambda -- is the current weight for the constraint

        Return values:
        lambda -- return the weight for the constraints
        """
        return current_lambda * 10

    def _scaled_sequential_penalty_increase(self, current_lambda: float, wu: float, t: int) -> float:
        """function for updating weights for constraints

        Keyword arguments:
        current_lambda -- is the current weight for the constraint
        wU -- is the upper bound of a valid penalty weight
        t -- int is the maximum number of iterations

        Return values:
        lambda -- return the weight for the constraints
        """
        scale_factor = wu ** (1 / t)
        return float(round(current_lambda * scale_factor))

    def _binary_search_penalty_algorithm(self, current_lambda: float, wu: float) -> float:
        """function for updating weights for constraints

        Keyword arguments:
        current_lambda -- is the current weight for the constraint
        wU -- is the upper bound of a valid penalty weight
        t -- int is the maximum number of iterations

        Return values:
        lambda -- return the weight for the constraints
        """
        return float(round(math.sqrt(current_lambda * wu)))
