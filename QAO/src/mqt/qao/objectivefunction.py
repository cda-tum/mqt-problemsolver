from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

# for managing symbols
from sympy import Expr, expand

if TYPE_CHECKING:
    from qubovert import PUBO

    from mqt.qao.variables import Variables


class ObjectiveFunction:
    """class for declaring and managing the objective function"""

    def __init__(self) -> None:
        """declaration of the list of objective functions and their corresponding weights"""
        self.objective_functions: list[tuple[Expr, float]] = []

    def add_objective_function(self, objective_function: Expr, minimization: bool = True, weight: float = 1) -> None:
        """function for adding an objective function in the objective functions list

        Keyword arguments:
        objective function -- Expression of the objective function --> For the current moment, the input expression is a polynomial but a future expansion will support also non-linear functions
        minimization -- boolean variable which specify if is a cost function of a fitness function. In the first case, the sign of the expression is changed. Set at True by default
        weights -- float for weighting the cost functions. Useful only in case of multi-objective optimization with a priori criteria


        Return values:
        None -- no output is expected
        """
        if not minimization:
            objective_function = -objective_function
        self.objective_functions.append((cast(Expr, expand(objective_function).evalf()), weight))  # type: ignore[no-untyped-call]

    def rewrite_cost_functions(self, pubo: PUBO, var: Variables) -> PUBO | bool:
        """function for rewriting the cost functions according with the variable structure

        Keyword arguments:
        pubo -- is the input pubo model
        var -- in the variables object


        Return values:
        qubo -- return the qubo model
        """
        for obj in self.objective_functions:
            obj_str = str(obj[0])
            if obj_str.startswith("-") and obj_str[1] != " ":
                obj_str = obj_str[0] + " " + obj_str[1:]
            fields = str(obj_str).replace("**", "^").split(" ")
            func = 0.0
            sign = "+"
            for field in fields:
                if field != "+" and field != "-":
                    poly_fields = field.split("*")
                    to_add = 1.0
                    for poly_field in poly_fields:
                        powers = poly_field.split("^")
                        temp = 0.0
                        if len(powers) == 2:
                            try:
                                power = int(powers[1])
                            except TypeError:
                                print("Expression not supported\n")
                                return False
                            key = powers[0]
                            if key not in var.binary_variables_name_weight:
                                print("Expression not supported\n")
                                return False
                            if isinstance(var.binary_variables_name_weight[key], list):
                                encoding = var.binary_variables_name_weight[key][0]
                                if encoding == "dictionary":
                                    for elem in var.binary_variables_name_weight[key]:
                                        if not isinstance(elem, str):
                                            t = 1.0
                                            t *= elem[0]
                                            if len(elem) == 2:
                                                t *= elem[1] ** power
                                            elif len(elem) == 3:
                                                t = t * elem[1] ** power + elem[2] ** power
                                            temp += t
                                else:
                                    for elem in var.binary_variables_name_weight[key]:
                                        if not isinstance(elem, str):
                                            t = 1.0
                                            t *= elem[0]
                                            if len(elem) == 2:
                                                t *= elem[1]
                                            elif len(elem) == 3:
                                                t = t * elem[1] + elem[2]
                                            temp += t
                                    temp = temp**power
                            elif isinstance(var.binary_variables_name_weight[key], tuple):
                                t = 1.0
                                t *= var.binary_variables_name_weight[key][0]
                                if len(var.binary_variables_name_weight[key]) == 2:
                                    t *= (var.binary_variables_name_weight[key][1]) ** power
                                elif len(var.binary_variables_name_weight[key]) == 3:
                                    t = (
                                        t * var.binary_variables_name_weight[key][1]
                                        + var.binary_variables_name_weight[key][2]
                                    ) ** power
                                temp += t
                            else:
                                temp = var.binary_variables_name_weight[key]
                            to_add *= temp
                        else:
                            key = poly_field
                            if key not in var.binary_variables_name_weight:
                                try:
                                    to_add *= float(key)
                                except TypeError:
                                    print("Expression not supported\n")
                                    return False
                            else:
                                if isinstance(var.binary_variables_name_weight[key], list):
                                    for elem in var.binary_variables_name_weight[key]:
                                        if not isinstance(elem, str):
                                            t = 1.0
                                            t *= elem[0]
                                            if len(elem) == 2:
                                                t *= elem[1]
                                            elif len(elem) == 3:
                                                t = t * elem[1] + elem[2]
                                            temp += t
                                elif isinstance(var.binary_variables_name_weight[key], tuple):
                                    t = 1.0
                                    t *= var.binary_variables_name_weight[key][0]
                                    if len(var.binary_variables_name_weight[key]) == 2:
                                        t *= var.binary_variables_name_weight[key][1]
                                    elif len(var.binary_variables_name_weight[key]) == 3:
                                        t = (
                                            t * var.binary_variables_name_weight[key][1]
                                            + var.binary_variables_name_weight[key][2]
                                        )
                                    temp += t
                                else:
                                    temp += var.binary_variables_name_weight[key]
                                to_add *= temp

                    if sign == "+":
                        func += to_add
                    else:
                        func -= to_add
                else:
                    sign = field
            pubo += func * obj[1]
        return pubo

    def substitute_values(self, solution: dict[str, Any], variables: Variables) -> dict[str, float] | bool:
        """function for substituting solutions into the cost functions expressions

        Keyword arguments:
        solution -- is the cost function solutions
        var -- in the variables object


        Return values:
        objective_functions_values -- values assumed by each objective function with the solution
        """
        objective_functions_values = {}
        for obj in self.objective_functions:
            objective_functions_values[str(obj[0])] = 0.0
            temp_expression = obj[0]
            for var in solution:
                if var in variables.variables_dict:
                    if isinstance(solution[var], float):
                        temp_expression = temp_expression.subs({variables.variables_dict[var].symbol: solution[var]})  # type: ignore[no-untyped-call]
                    elif isinstance(solution[var], list):
                        for i in range(len(solution[var])):
                            if isinstance(solution[var][i], list):
                                for j in range(len(solution[var][i])):
                                    if isinstance(solution[var][i][j], list):
                                        for k in range(len(solution[var][i][j])):
                                            temp_expression = temp_expression.subs(  # type: ignore[no-untyped-call]
                                                {variables.variables_dict[var][i][j][k].symbol: solution[var][i][j][k]}
                                            )
                                    else:
                                        temp_expression = temp_expression.subs(  # type: ignore[no-untyped-call]
                                            {variables.variables_dict[var][i][j].symbol: solution[var][i][j]}
                                        )
                            else:
                                temp_expression = temp_expression.subs(  # type: ignore[no-untyped-call]
                                    {variables.variables_dict[var][i].symbol: solution[var][i]}
                                )
            try:
                objective_functions_values[str(obj[0])] = float(temp_expression)
            except ValueError:
                return False

        return objective_functions_values
