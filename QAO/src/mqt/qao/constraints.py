from __future__ import annotations

from typing import Any, cast

from qubovert import PUBO, boolean_var

# for managing symbols
from sympy import Expr, expand

from .variables import Variable, Variables


class Constraints:
    """class for declaring and managing constraints"""

    def __init__(self) -> None:
        """declaration of the list of penalty functions and the list of constraint"""
        self.constraints_penalty_functions: list[tuple[PUBO, bool]] = []
        self.constraints: list[tuple[str, bool, bool, bool]] = []

    def add_constraint_penalty_function(self, constraint: PUBO, hard: bool = True) -> None:
        """function for adding a constraint.

        Keyword arguments:
        constraint -- it is the constraint that we want to add
        hard -- is boolean variable which said if the constraint is hard or weak

        Return values:
        None
        """
        self.constraints_penalty_functions.append((constraint, hard))

    def add_constraint(
        self, expression: str, hard: bool = True, to_substitute: bool = True, variable_precision: bool = True
    ) -> None:
        """function for adding a constraint.

        Keyword arguments:
        Expression -- Expr is the constraint expression
        hard -- bool saying if the constraint is hard or weak
        to_substitute -- bool saying if the variable of the expression are already binary object of pyqubo, or are element of the variable class

        Return values:
        None
        """
        self.constraints.append((expression, hard, to_substitute, variable_precision))

    def translate_constraints(self, var: Variables) -> tuple[bool, Variables]:
        """function for translating the constraint into penalty functions.

        Keyword arguments:
        var -- problem variables

        Return values:
        bool -- for identifying eventual errors or issue in the conversion
        auxiliary_variables -- variable added for writing constraints
        """
        self.constraints_penalty_functions = []
        auxiliary_variables = Variables()
        i = 0
        j = 0
        for elem in self.constraints:
            string = elem[0]
            if "~" in string:
                # Not constraints
                constraint = self._not_constraint(elem, var)
            elif "&" in string:
                # And constraints
                constraint = self._and_constraint(elem, var)
            elif "|" in string:
                # Or constraints
                constraint = self._or_constraint(elem, var)
            elif "^" in string:
                # Xor constraints
                constraint = self._xor_constraint(elem, var)
            elif ">=" in string:
                # greater equal constraint
                constraint, i, j, auxiliary_variables = self._greq_constraint(elem, var, auxiliary_variables, i, j)
            elif "<=" in string:
                # less equal constraint
                constraint, i, j, auxiliary_variables = self._lteq_constraint(elem, var, auxiliary_variables, i, j)
            elif ">" in string:
                # greater constraint
                constraint, i, j, auxiliary_variables = self._grt_constraint(elem, var, auxiliary_variables, i, j)
            elif "<" in string:
                # less constraint
                constraint, i, j, auxiliary_variables = self._lt_constraint(elem, var, auxiliary_variables, i, j)
            elif "=" in string:
                # equal constraint
                constraint = self._eq_constraint(elem, var)
            else:
                # Not supported constraints
                print("Not supported type of constraints\n")
                return False, auxiliary_variables
            self.add_constraint_penalty_function(constraint, elem[1])
        return True, auxiliary_variables

    def check_constraint(
        self,
        solution: dict[str, Any],
        solution_original_variables: dict[str, int],
        variables: Variables,
        weak: bool = False,
    ) -> tuple[bool, list[bool]]:
        """function for checking if the constraints are satisfied for a given solution.

        Keyword arguments:
        solution -- is the dictionary containing the problem solution to considered
        variables -- variables of the problem of interest
        all_satisfied -- a boolean variable saying if all the constraints are satisfied
        satisfied_single -- a list of bool saying constraint by constraint if it is satisfied or not

        Return values:
        all_satisfied -- a boolean variable saying if all the constraints are satisfied
        satisfied_single -- a list of bool saying constraint by constraint if it is satisfied or not
        """
        all_satisfied = True
        satisfied = False
        single_satisfied = []
        for elem in self.constraints:
            string = elem[0]
            if elem[1] or not weak:
                if "~" in string:
                    # Not constraints
                    if elem[2]:
                        satisfied = self._not_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._not_constraint_check_no_sub(string, solution_original_variables)
                elif "&" in string:
                    # And constraints
                    if elem[2]:
                        satisfied = self._and_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._and_constraint_check_no_sub(string, solution_original_variables)
                elif "|" in string:
                    # Or constraints
                    if elem[2]:
                        satisfied = self._or_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._or_constraint_check_no_sub(string, solution_original_variables)
                elif "^" in string:
                    # Xor constraints
                    if elem[2]:
                        satisfied = self._xor_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._xor_constraint_check_no_sub(string, solution_original_variables)

                elif ">=" in string:
                    # greater equal constraint
                    if elem[2]:
                        satisfied = self._greq_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._greq_constraint_check_no_sub(string, solution_original_variables)

                elif "<=" in string:
                    # less equal constraint
                    if elem[2]:
                        satisfied = self._lteq_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._lteq_constraint_check_no_sub(string, solution_original_variables)

                elif ">" in string:
                    # greater constraint
                    if elem[2]:
                        satisfied = self._grt_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._grt_constraint_check_no_sub(string, solution_original_variables)

                elif "<" in string:
                    # less constraint
                    if elem[2]:
                        satisfied = self._lt_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._lt_constraint_check_no_sub(string, solution_original_variables)

                elif "=" in string:
                    # equal constraint
                    if elem[2]:
                        satisfied = self._eq_constraint_check(string, variables, solution)
                    else:
                        satisfied = self._eq_constraint_check_no_sub(string, solution_original_variables)
            if not satisfied:
                all_satisfied = False
            single_satisfied.append(satisfied)
        return all_satisfied, single_satisfied

    def _not_constraint(self, elem: tuple[str, bool, bool, bool], var: Variables) -> PUBO | bool:
        """function for writing the not constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables

        Return values:
        Constraint -- penalty function to add
        or bool -- in case of issues

        It exploits the automatic pyqubo conversion
        Not(a) = b --> 2ab - a - b + 1
        """
        ret: PUBO | bool = PUBO()
        if elem[2]:
            el = (elem[0]).replace("~", "").split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            a = self._convert_expression_logic(
                expand(cast("Expr", (el[0]))).evalf(),
                var.binary_variables_name_weight,
            )
            b = self._convert_expression_logic(
                expand(cast("Expr", (el[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            ret = 2 * a * b - a - b + 1
        else:
            el = (elem[0]).replace("~", "").replace(" ", "").split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False

            op = self._logic_constraint(el, var)

            ret = 2 * op[0] * op[1] - op[0] - op[1] + 1
        return ret

    def _not_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the not constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).replace("~", "").split("=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, values in solution.items():
                if var in variables.variables_dict:
                    if isinstance(values, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, values, variables.variables_dict[var]
                        )
                    elif isinstance(values, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution,
                            variables,
                            expr1,
                            expr2,
                            var,
                            expr1_to_sub,
                            expr2_to_sub,
                        )
        try:
            return bool(expr1) != bool(expr2)
        except ValueError:
            return False

    @staticmethod
    def _not_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the not constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).replace("~", "").split("=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return bool(expr1) != bool(expr2)
        except ValueError:
            return False

    def _and_constraint(self, elem: tuple[str, bool, bool, bool], var: Variables) -> PUBO | bool:
        """function for writing the and constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables

        Return values:
        Constraint -- penalty function to add
        or bool -- in case of issues

        It exploits the automatic pyqubo conversion
        a and b = c --> ab -2(a+b)c + 3c
        """
        if elem[2]:
            el = str(elem[0]).split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("&")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False
            a = self._convert_expression_logic(
                expand(cast("Expr", (el2[0]))).evalf(),
                var.binary_variables_name_weight,
            )
            b = self._convert_expression_logic(
                expand(cast("Expr", (el2[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            c = self._convert_expression_logic(
                expand(cast("Expr", (el[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            ret = a * b - 2 * (a + b) * c + 3 * c
        else:
            el = str(elem[0]).replace(" ", "").split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("&")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False

            el = [el2[0], el2[1], el[1]]
            op = self._logic_constraint(el, var)

            ret = op[0] * op[1] - 2 * (op[0] + op[1]) * op[2] + 3 * op[2]
        return ret

    def _and_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the and constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("&")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        expr3_to_sub = bool(not str(expr3).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub or expr3_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                            expr1_to_sub,
                            expr2_to_sub,
                            expr3_to_sub,
                            expr1,
                            expr2,
                            expr3,
                            value,
                            variables.variables_dict[var],
                        )
                    elif isinstance(value, list):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_list_var(
                            solution, variables, expr1, expr2, expr3, var, expr1_to_sub, expr2_to_sub, expr3_to_sub
                        )
        try:
            return bool(expr1) and bool(expr2) == bool(expr3)
        except ValueError:
            return False

    @staticmethod
    def _and_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the and constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("&")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        expr3_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression3 = expr3.free_symbols
        else:
            symbols_in_the_expression3 = set()
            expr3_to_sub = False
        temp = set(list(symbols_in_the_expression1) + list(symbols_in_the_expression2 - symbols_in_the_expression1))
        symbols_in_the_expression = list(symbols_in_the_expression3) + list(symbols_in_the_expression3 - temp)
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
                if expr3_to_sub:
                    expr3 = expr3.subs({symbol: solution[var]})
        try:
            return bool(expr1) and bool(expr2) == bool(expr3)
        except ValueError:
            return False

    def _or_constraint(self, elem: tuple[str, bool, bool, bool], var: Variables) -> PUBO | bool:
        """function for writing the or constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables

        Return values:
        Constraint -- penalty function to add
        or bool -- in case of issues

        it exploits the automatic pybo conversion
        a or b = c --> ab + (a+b)(1-2c) + c
        """
        if elem[2]:
            el = str(elem[0]).split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("|")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False
            a = self._convert_expression_logic(
                expand(cast("Expr", (el2[0]))).evalf(),
                var.binary_variables_name_weight,
            )
            b = self._convert_expression_logic(
                expand(cast("Expr", (el2[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            c = self._convert_expression_logic(
                expand(cast("Expr", (el[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            ret = a * b + (a + b) * (1 - 2 * c) + c
        else:
            el = str(elem[0]).replace(" ", "").split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("|")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False

            el = [el2[0], el2[1], el[1]]
            op = self._logic_constraint(el, var)
            ret = op[0] * op[1] + (op[0] + op[1]) * (1 - 2 * op[2]) + op[2]
        return ret

    def _or_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the or constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("|")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        expr3_to_sub = bool(not str(expr3).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub or expr3_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                            expr1_to_sub,
                            expr2_to_sub,
                            expr3_to_sub,
                            expr1,
                            expr2,
                            expr3,
                            value,
                            variables.variables_dict[var],
                        )
                    elif isinstance(value, list):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_list_var(
                            solution, variables, expr1, expr2, expr3, var, expr1_to_sub, expr2_to_sub, expr3_to_sub
                        )
        try:
            return bool(expr1) or bool(expr2) == bool(expr3)
        except ValueError:
            return False

    @staticmethod
    def _or_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the or constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("|")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        expr3_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression3 = expr3.free_symbols
        else:
            symbols_in_the_expression3 = set()
            expr3_to_sub = False
        temp = set(list(symbols_in_the_expression1) + list(symbols_in_the_expression2 - symbols_in_the_expression1))
        symbols_in_the_expression = list(symbols_in_the_expression3) + list(symbols_in_the_expression3 - temp)
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
                if expr3_to_sub:
                    expr3 = expr3.subs({symbol: solution[var]})
        try:
            return bool(expr1) or bool(expr2) == bool(expr3)
        except ValueError:
            return False

    @staticmethod
    def _logic_constraint(el: list[str], var: Variables) -> list[PUBO]:
        op = []
        for e in el:
            for elms in var.binary_variables_name_weight.values():
                if isinstance(elms, list):
                    for elm in elms:
                        if not isinstance(elm, str) and e == next(iter(elm[0].variables)):
                            op.append(elm[0])  # noqa: PERF401
                elif e == next(iter(elms[0].variables)):
                    op.append(elms[0])
        return op

    def _xor_constraint(self, elem: tuple[str, bool, bool, bool], var: Variables) -> PUBO | bool:
        """function for writing the xor constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables

        Return values:
        Constraint -- penalty function to add

        My conversion
        a xor b = c --> abc + (1-a)(1-b)c + (1-a)b(1-c)+ a(1-b)(1-c) = 4abc + a + b + c - 2ac - 2bc - 2ab
        """
        if elem[2]:
            el = str(elem[0]).split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("^")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False
            a = self._convert_expression_logic(expand(cast("Expr", (el2[0]))).evalf(), var.binary_variables_name_weight)
            b = self._convert_expression_logic(expand(cast("Expr", (el2[1]))).evalf(), var.binary_variables_name_weight)
            c = self._convert_expression_logic(expand(cast("Expr", (el[1]))).evalf(), var.binary_variables_name_weight)
            ret = a + b + c - 2 * a * c - 2 * b * c - 2 * a * b + 4 * a * b * c
        else:
            el = str(elem[0]).replace(" ", "").split("=")
            if len(el) != 2:
                print("Wrong constraint format\n")
                return False
            el2 = el[0].split("^")
            if len(el2) != 2:
                print("Wrong constraint format\n")
                return False
            el = [el2[0], el2[1], el[1]]
            op = self._logic_constraint(el, var)
            a = op[0]
            b = op[1]
            c = op[2]
            ret = a + b + c - 2 * a * c - 2 * b * c - 2 * a * b + 4 * a * b * c
        return ret

    @staticmethod
    def _constraint_check_three_sub_single_var(
        expr1_to_sub: bool,
        expr2_to_sub: bool,
        expr3_to_sub: bool,
        expr1: Expr,
        expr2: Expr,
        expr3: Expr,
        sol_val: float,
        variable: Variable,
    ) -> tuple[Expr, Expr, Expr]:
        if expr1_to_sub:
            expr1 = expr1.subs({variable.symbol: sol_val})
        if expr2_to_sub:
            expr2 = expr2.subs({variable.symbol: sol_val})
        if expr3_to_sub:
            expr3 = expr3.subs({variable.symbol: sol_val})
        return expr1, expr2, expr3

    def _constraint_check_three_sub_list_var(
        self,
        solution: dict[str, Any],
        variables: Variables,
        expr1: Expr,
        expr2: Expr,
        expr3: Expr,
        var: str,
        expr1_to_sub: bool,
        expr2_to_sub: bool,
        expr3_to_sub: bool,
    ) -> tuple[Expr, Expr, Expr]:
        for i in range(len(solution[var])):
            if isinstance(solution[var][i], list):
                for j in range(len(solution[var][i])):
                    if isinstance(solution[var][i][j], list):
                        for k in range(len(solution[var][i][j])):
                            expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                                expr1_to_sub,
                                expr2_to_sub,
                                expr3_to_sub,
                                expr1,
                                expr2,
                                expr3,
                                solution[var][i][j][k],
                                variables.variables_dict[var][i][j][k],
                            )
                    else:
                        expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                            expr1_to_sub,
                            expr2_to_sub,
                            expr3_to_sub,
                            expr1,
                            expr2,
                            expr3,
                            solution[var][i][j],
                            variables.variables_dict[var][i][j],
                        )
            else:
                expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                    expr1_to_sub,
                    expr2_to_sub,
                    expr3_to_sub,
                    expr1,
                    expr2,
                    expr3,
                    solution[var][i],
                    variables.variables_dict[var][i],
                )
        return expr1, expr2, expr3

    def _xor_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the xor constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("^")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        expr3_to_sub = bool(not str(expr3).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub or expr3_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_single_var(
                            expr1_to_sub,
                            expr2_to_sub,
                            expr3_to_sub,
                            expr1,
                            expr2,
                            expr3,
                            value,
                            variables.variables_dict[var],
                        )
                    elif isinstance(value, list):
                        expr1, expr2, expr3 = self._constraint_check_three_sub_list_var(
                            solution, variables, expr1, expr2, expr3, var, expr1_to_sub, expr2_to_sub, expr3_to_sub
                        )
        try:
            return bool(expr1) ^ bool(expr2) == bool(expr3)
        except ValueError:
            return False

    @staticmethod
    def _xor_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the xor constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = str(constraint).split("=")
        if len(el) != 2:
            print("Wrong constraint format\n")
            return False
        el2 = el[0].split("^")
        if len(el2) != 2:
            print("Wrong constraint format\n")
            return False
        expr1 = expand(cast("Expr", (el2[0]))).evalf()
        expr2 = expand(cast("Expr", (el2[1]))).evalf()
        expr3 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        expr3_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression3 = expr3.free_symbols
        else:
            symbols_in_the_expression3 = set()
            expr3_to_sub = False
        temp = set(list(symbols_in_the_expression1) + list(symbols_in_the_expression2 - symbols_in_the_expression1))
        symbols_in_the_expression = list(symbols_in_the_expression3) + list(symbols_in_the_expression3 - temp)
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
                if expr3_to_sub:
                    expr3 = expr3.subs({symbol: solution[var]})
        try:
            return bool(expr1) ^ bool(expr2) == bool(expr3)
        except ValueError:
            return False

    @staticmethod
    def _constraint_check_sub_single_var(
        expr1_to_sub: bool, expr2_to_sub: bool, expr1: Expr, expr2: Expr, sol_val: float, variable: Variable
    ) -> tuple[Expr, Expr]:
        if expr1_to_sub:
            expr1 = expr1.subs({variable.symbol: sol_val})
        if expr2_to_sub:
            expr2 = expr2.subs({variable.symbol: sol_val})
        return expr1, expr2

    def _constraint_check_sub_list_var(
        self,
        solution: dict[str, Any],
        variables: Variables,
        expr1: Expr,
        expr2: Expr,
        var: str,
        expr1_to_sub: bool,
        expr2_to_sub: bool,
    ) -> tuple[Expr, Expr]:
        for i in range(len(solution[var])):
            if isinstance(solution[var][i], list):
                for j in range(len(solution[var][i])):
                    if isinstance(solution[var][i][j], list):
                        for k in range(len(solution[var][i][j])):
                            expr1, expr2 = self._constraint_check_sub_single_var(
                                expr1_to_sub,
                                expr2_to_sub,
                                expr1,
                                expr2,
                                solution[var][i][j][k],
                                variables.variables_dict[var][i][j][k],
                            )
                    else:
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub,
                            expr2_to_sub,
                            expr1,
                            expr2,
                            solution[var][i][j],
                            variables.variables_dict[var][i][j],
                        )
            else:
                expr1, expr2 = self._constraint_check_sub_single_var(
                    expr1_to_sub,
                    expr2_to_sub,
                    expr1,
                    expr2,
                    solution[var][i],
                    variables.variables_dict[var][i],
                )
        return expr1, expr2

    def _greq_constraint(
        self, elem: tuple[str, bool, bool, bool], var: Variables, aux: Variables, i: int, j: int
    ) -> tuple[
        PUBO,
        int,
        int,
        Variables,
    ]:
        """function for writing the grater equal  constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables
        aux -- auxiliry variable object for adding auxiliary variable
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable

        Return values:
        Constraint -- penalty function to add
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable
        aux -- auxiliry variable object for adding auxiliary variable
        or bool -- in case of issues

        My conversion
        sum >= b--> sum-a = b with a assuming the range max(sum)-b, 0 --> (sum-a-b)^2
        """
        ret: tuple[PUBO, int, int, Variables]
        if elem[2]:
            el = str(elem[0]).split(">=")
            var_precision = 1.0
            exp = PUBO()
            if elem[3]:
                ret_func = self._convert_expression_prec(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                    var.variables_dict,
                )
                if isinstance(ret_func, tuple):
                    exp = ret_func[0]
                    var_precision = ret_func[1]
            else:
                exp = self._convert_expression(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                )

            _min_val, max_val, const = self._min_max_const_estimation(exp)
            if max_val + const > 0:
                if len(str(max_val + const).split(".")) == 2:
                    if elem[3] and float("0." + str(max_val + const).split(".")[1]) != 0.0:
                        precision = min(float("0." + str(max_val + const).split(".")[1]), var_precision)
                    elif elem[3]:
                        precision = var_precision
                    else:
                        precision = float("0." + str(max_val + const).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = min(1, var_precision) if elem[3] else 1
                aux_var = aux.add_continuous_variable("aux" + format(j), 0, max_val + const, precision)
                self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                exp_auxiliary = self._convert_expression(cast("Expr", (aux_var)), aux.binary_variables_name_weight)
                j += 1
                ret = (exp - exp_auxiliary) ** 2, i, j, aux
            else:
                ret = (exp) ** 2, i, j, aux
        else:
            el = str(elem[0]).split(">=")
            exp = expand(cast("Expr", (el[0] + "-" + el[1]))).evalf()
            _min_val, max_val, const = self._min_max_const_estimation(exp)
            if max_val + const > 0:
                if len(str(max_val + const).split(".")) == 2:
                    precision = float("0." + str(max_val + const).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = 1
                aux_var = aux.add_continuous_variable("aux" + format(j), 0, max_val + const, precision)
                j += 1
                self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                exp_auxiliary = self._convert_expression(cast("Expr", (aux_var)), aux.binary_variables_name_weight)
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp - exp_auxiliary),
                    list(var.binary_variables_name_weight.values()) + list(aux.binary_variables_name_weight.values()),
                )
                ret = hamiltonian**2, i, j, aux
            else:
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp), list(var.binary_variables_name_weight.values())
                )
                ret = hamiltonian**2, i, j, aux

        return ret

    def _greq_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the grater equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = constraint.split(">=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, value, variables.variables_dict[var]
                        )
                    elif isinstance(value, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution, variables, expr1, expr2, var, expr1_to_sub, expr2_to_sub
                        )
        try:
            return float(expr1) >= float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _greq_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the grater equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).split(">=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return float(expr1) >= float(expr2)
        except ValueError:
            return False

    def _lteq_constraint(
        self, elem: tuple[str, bool, bool, bool], var: Variables, aux: Variables, i: int, j: int
    ) -> tuple[
        PUBO,
        int,
        int,
        Variables,
    ]:
        """function for writing the less equal  constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables
        aux -- auxiliry variable object for adding auxiliary variable
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable

        Return values:
        Constraint -- penalty function to add
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable
        aux -- auxiliry variable object for adding auxiliary variable

        My conversion
        sum <= b--> sum+a = b with a assuming the range b-min(sum), 0 --> (sum+a-b)^2
        """
        ret: tuple[PUBO, int, int, Variables]
        if elem[2]:
            el = str(elem[0]).split("<=")
            var_precision = 1.0
            exp = PUBO()
            if elem[3]:
                ret_func = self._convert_expression_prec(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                    var.variables_dict,
                )
                if isinstance(ret_func, tuple):
                    exp = ret_func[0]
                    var_precision = ret_func[1]
            else:
                exp = self._convert_expression(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                )
            min_val, _max_val, const = self._min_max_const_estimation(exp)
            const = -const
            if const - min_val > 0:
                if len(str(const - min_val).split(".")) == 2:
                    if elem[3] and float("0." + str(const - min_val).split(".")[1]) != 0.0:
                        precision = min(float("0." + str(const - min_val).split(".")[1]), var_precision)
                    elif elem[3]:
                        precision = var_precision
                    else:
                        precision = float("0." + str(const - min_val).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = min(1, var_precision) if elem[3] else 1
                aux_var = aux.add_continuous_variable("aux" + format(j), 0, const - min_val, precision)
                j += 1
                self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                exp_auxiliary = self._convert_expression((cast("Expr", (aux_var))), aux.binary_variables_name_weight)
                ret = (exp + exp_auxiliary) ** 2, i, j, aux
            else:
                ret = (exp) ** 2, i, j, aux
        else:
            el = str(elem[0]).split("<=")
            exp = expand(cast("Expr", (el[0] + "-" + el[1]))).evalf()
            min_val, _max_val, const = self._min_max_const_estimation(exp)
            const = -const
            if const - min_val > 0:
                if len(str(const - min_val).split(".")) == 2:
                    precision = float("0." + str(const - min_val).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = 1
                aux_var = aux.add_continuous_variable("aux" + format(j), 0, const - min_val, precision)
                j += 1
                self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                exp_auxiliary = self._convert_expression((cast("Expr", (aux_var))), aux.binary_variables_name_weight)
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp + exp_auxiliary),
                    list(var.binary_variables_name_weight.values()) + list(aux.binary_variables_name_weight.values()),
                )
                ret = hamiltonian**2, i, j, aux
            else:
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp), list(var.binary_variables_name_weight.values())
                )
                ret = hamiltonian**2, i, j, aux
        return ret

    def _lteq_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the less equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = constraint.split("<=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, value, variables.variables_dict[var]
                        )
                    elif isinstance(value, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution, variables, expr1, expr2, var, expr1_to_sub, expr2_to_sub
                        )
        try:
            return float(expr1) <= float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _lteq_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the less equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).split("<=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return float(expr1) <= float(expr2)
        except ValueError:
            return False

    def _grt_constraint(
        self, elem: tuple[str, bool, bool, bool], var: Variables, aux: Variables, i: int, j: int
    ) -> tuple[
        PUBO,
        int,
        int,
        Variables,
    ]:
        """function for writing the greater constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables
        aux -- auxiliry variable object for adding auxiliary variable
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable

        Return values:
        Constraint -- penalty function to add
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable
        aux -- auxiliry variable object for adding auxiliary variable

        My conversion
        sum > b--> sum-a = b with a assuming the range max(sum)-b, precision --> (sum-a-b)^2
        """
        ret: tuple[PUBO, int, int, Variables]
        if elem[2]:
            el = str(elem[0]).split(">")
            var_precision = 1.0
            exp = PUBO()
            if elem[3]:
                ret_func = self._convert_expression_prec(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                    var.variables_dict,
                )
                if isinstance(ret_func, tuple):
                    exp = ret_func[0]
                    var_precision = ret_func[1]
            else:
                exp = self._convert_expression(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                )
            _min_val, max_val, const = self._min_max_const_estimation(exp)
            if max_val + const > 0:
                if len(str(max_val + const).split(".")) == 2:
                    if elem[3] and float("0." + str(max_val + const).split(".")[1]) != 0.0:
                        precision = min(float("0." + str(max_val + const).split(".")[1]), var_precision)
                    elif elem[3]:
                        precision = var_precision
                    else:
                        precision = float("0." + str(max_val + const).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = min(1, var_precision) if elem[3] else 1
                if precision != max_val + const:
                    aux_var = aux.add_continuous_variable("aux" + format(j), precision, max_val + const, precision)
                    j += 1
                    self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                    exp_auxiliary = self._convert_expression(
                        (cast("Expr", (aux_var))), aux.binary_variables_name_weight
                    )
                    ret = (exp - exp_auxiliary) ** 2, i, j, aux
                else:
                    ret = (exp - precision) ** 2, i, j, aux
            else:
                ret = (exp) ** 2, i, j, aux
        else:
            el = str(elem[0]).split(">")
            exp = expand(cast("Expr", (el[0] + "-" + el[1]))).evalf()
            _min_val, max_val, const = self._min_max_const_estimation(exp)
            if max_val + const > 0:
                if len(str(max_val + const).split(".")) == 2:
                    precision = float("0." + str(max_val + const).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = 1
                if precision != max_val + const:
                    aux_var = aux.add_continuous_variable("aux" + format(j), precision, max_val + const, precision)
                    j += 1
                    self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                    exp_auxiliary = self._convert_expression(
                        (cast("Expr", (aux_var))), aux.binary_variables_name_weight
                    )
                    hamiltonian = self._expression_to_hamiltonian(
                        cast("Expr", exp - exp_auxiliary),
                        list(var.binary_variables_name_weight.values())
                        + list(aux.binary_variables_name_weight.values()),
                    )
                    ret = hamiltonian**2, i, j, aux
                else:
                    hamiltonian = self._expression_to_hamiltonian(
                        cast("Expr", exp - precision),
                        list(var.binary_variables_name_weight.values())
                        + list(aux.binary_variables_name_weight.values()),
                    )
                    ret = hamiltonian**2, i, j, aux
            else:
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp), list(var.binary_variables_name_weight.values())
                )
                ret = hamiltonian**2, i, j, aux
        return ret

    def _grt_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the greter constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = constraint.split(">")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, value, variables.variables_dict[var]
                        )
                    elif isinstance(value, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution, variables, expr1, expr2, var, expr1_to_sub, expr2_to_sub
                        )
        try:
            return float(expr1) > float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _grt_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the greater equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).split(">")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return float(expr1) > float(expr2)
        except ValueError:
            return False

    def _lt_constraint(
        self, elem: tuple[str, bool, bool, bool], var: Variables, aux: Variables, i: int, j: int
    ) -> tuple[
        PUBO,
        int,
        int,
        Variables,
    ]:
        """function for writing the less constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables
        aux -- auxiliry variable object for adding auxiliary variable
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable

        Return values:
        Constraint -- penalty function to add
        i -- variable for counting binaryy expansion of the auxiliary variable
        j -- variable for counting the auxiliary variable
        aux -- auxiliry variable object for adding auxiliary variable

        My conversion
        sum < b--> sum+a = b with a assuming the range b-min(sum), precision --> (sum+a-b)^2
        """
        ret: tuple[PUBO, int, int, Variables]
        if elem[2]:
            el = str(elem[0]).split("<")
            var_precision = 1.0
            exp = PUBO()
            if elem[3]:
                ret_func = self._convert_expression_prec(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                    var.variables_dict,
                )
                if isinstance(ret_func, tuple):
                    exp = ret_func[0]
                    var_precision = ret_func[1]
            else:
                exp = self._convert_expression(
                    expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                    var.binary_variables_name_weight,
                )
            min_val, _max_val, const = self._min_max_const_estimation(exp)
            precision = 1.0
            const = -const
            if const - min_val > 0:
                if len(str(const - min_val).split(".")) == 2:
                    if elem[3] and float("0." + str(const - min_val).split(".")[1]) != 0.0:
                        precision = min(float("0." + str(const - min_val).split(".")[1]), var_precision)
                    elif elem[3]:
                        precision = var_precision
                    else:
                        precision = float("0." + str(const - min_val).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                elif elem[3]:
                    precision = min(1, var_precision)
                if precision != const - min_val:
                    aux_var = aux.add_continuous_variable("aux" + format(j), precision, const - min_val, precision)
                    j += 1
                    self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                    exp_auxiliary = self._convert_expression(
                        (cast("Expr", (aux_var))), aux.binary_variables_name_weight
                    )
                    ret = (exp + exp_auxiliary) ** 2, i, j, aux
                else:
                    ret = (exp + precision) ** 2, i, j, aux
            else:
                ret = (exp) ** 2, i, j, aux
        else:
            el = str(elem[0]).split("<")
            exp = expand(cast("Expr", (el[0] + "-" + el[1]))).evalf()
            min_val, _max_val, const = self._min_max_const_estimation(exp)
            const = -const
            if const - min_val > 0:
                if len(str(const - min_val).split(".")) == 2:
                    precision = float("0." + str(const - min_val).split(".")[1])
                    if precision == 0.0:
                        precision = 1
                else:
                    precision = 1
                if precision != const - min_val:
                    aux_var = aux.add_continuous_variable("aux" + format(j), precision, const - min_val, precision)
                    j += 1
                    self.constraints, i = aux.move_to_binary(self.constraints, i, "__a")
                    exp_auxiliary = self._convert_expression(
                        (cast("Expr", (aux_var))), aux.binary_variables_name_weight
                    )
                    hamiltonian = self._expression_to_hamiltonian(
                        cast("Expr", exp + exp_auxiliary),
                        list(var.binary_variables_name_weight.values())
                        + list(aux.binary_variables_name_weight.values()),
                    )
                    ret = hamiltonian**2, i, j, aux
                else:
                    hamiltonian = self._expression_to_hamiltonian(
                        cast("Expr", exp + precision),
                        list(var.binary_variables_name_weight.values())
                        + list(aux.binary_variables_name_weight.values()),
                    )
                    ret = hamiltonian**2, i, j, aux
            else:
                hamiltonian = self._expression_to_hamiltonian(
                    cast("Expr", exp), list(var.binary_variables_name_weight.values())
                )
                ret = hamiltonian**2, i, j, aux
        return ret

    def _lt_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the less constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = constraint.split("<")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, value, variables.variables_dict[var]
                        )
                    elif isinstance(value, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution, variables, expr1, expr2, var, expr1_to_sub, expr2_to_sub
                        )
        try:
            return float(expr1) < float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _lt_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the less equal constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).split("<")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return float(expr1) < float(expr2)
        except ValueError:
            return False

    def _eq_constraint(self, elem: tuple[str, bool, bool, bool], var: Variables) -> PUBO | bool:
        """function for writing the less constraint penalty function.

        Keyword arguments:
        elem  -- tuples containing the constraints expression, a boolean variable which says if the expression is already written with Binary object of qubovert and one expressing the hardness of the constraint
        var -- problem variables

        Return values:
        Constraint -- penalty function to add
        or bool -- in case of issues

        My conversion
        sum = b--> (sum-b)^2
        """
        if elem[2]:
            el = str(elem[0]).split("=")
            exp = self._convert_expression(
                expand(cast("Expr", (el[0] + "-" + el[1]))).evalf(),
                var.binary_variables_name_weight,
            )
            ret = (exp) ** 2
        else:
            el = str(elem[0]).split("=")
            exp = expand(cast("Expr", (el[0] + "-" + el[1]))).evalf()
            hamiltonian = self._expression_to_hamiltonian(
                cast("Expr", exp), list(var.binary_variables_name_weight.values())
            )
            ret = hamiltonian**2
        return ret

    def _eq_constraint_check(self, constraint: str, variables: Variables, solution: dict[str, Any]) -> bool:
        """function for checking the equality constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        variables -- problem variables
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = constraint.split("=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = bool(not str(expr1).replace(".", "").isnumeric())
        expr2_to_sub = bool(not str(expr2).replace(".", "").isnumeric())
        if expr2_to_sub or expr1_to_sub:
            for var, value in solution.items():
                if var in variables.variables_dict:
                    if isinstance(value, float):
                        expr1, expr2 = self._constraint_check_sub_single_var(
                            expr1_to_sub, expr2_to_sub, expr1, expr2, value, variables.variables_dict[var]
                        )
                    elif isinstance(value, list):
                        expr1, expr2 = self._constraint_check_sub_list_var(
                            solution, variables, expr1, expr2, var, expr1_to_sub, expr2_to_sub
                        )
        try:
            return float(expr1) == float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _eq_constraint_check_no_sub(constraint: str, solution: dict[str, int]) -> bool:
        """function for checking the equality constraint penalty function.

        Keyword arguments:
        constraint -- string containing the constraint expression
        solution -- problem solution to evaluate

        Return values:
        bool -- saying if the constraint is satisfied or not
        """
        el = (constraint).split("=")
        expr1 = expand(cast("Expr", (el[0]))).evalf()
        expr2 = expand(cast("Expr", (el[1]))).evalf()
        expr1_to_sub = True
        if not str(expr1).replace(".", "").isnumeric():
            symbols_in_the_expression1 = expr1.free_symbols
        else:
            symbols_in_the_expression1 = set()
            expr1_to_sub = False
        expr2_to_sub = True
        if not str(expr2).replace(".", "").isnumeric():
            symbols_in_the_expression2 = expr2.free_symbols
        else:
            symbols_in_the_expression2 = set()
            expr2_to_sub = False
        symbols_in_the_expression = list(symbols_in_the_expression1) + list(
            symbols_in_the_expression2 - symbols_in_the_expression1
        )
        for symbol in symbols_in_the_expression:
            var = str(symbol)
            if var in solution and isinstance(solution[var], float):
                if expr1_to_sub:
                    expr1 = expr1.subs({symbol: solution[var]})
                if expr2_to_sub:
                    expr2 = expr2.subs({symbol: solution[var]})
        try:
            return float(expr1) == float(expr2)
        except ValueError:
            return False

    @staticmethod
    def _convert_expression_power(
        powers: list[str], binary_variables_name_weight: dict[str, Any], to_add: float
    ) -> float | bool:
        temp = 0.0
        try:
            power = int(powers[1])
        except TypeError:
            print("Expression not supported\n")
            return False
        key = powers[0]
        if key not in binary_variables_name_weight:
            print("Expression not supported\n")
            return False
        if isinstance(binary_variables_name_weight[key], list):
            encoding = binary_variables_name_weight[key][0]
            if encoding == "dictionary":
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1] ** power
                        elif len(elem) == 3:
                            t = t * elem[1] ** power + elem[2] ** power
                        temp += t
            else:
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1]
                        elif len(elem) == 3:
                            t = t * elem[1] + elem[2]
                        temp += t
                temp **= power
        else:
            t = 1.0
            t *= binary_variables_name_weight[key][0]
            if len(binary_variables_name_weight[key]) == 2:
                t *= (binary_variables_name_weight[key][1]) ** power
            elif len(binary_variables_name_weight[key]) == 3:
                t = (t * binary_variables_name_weight[key][1] + binary_variables_name_weight[key][2]) ** power
            temp += t
        to_add *= temp
        return to_add

    @staticmethod
    def _convert_expression_no_power(
        poly_field: str, binary_variables_name_weight: dict[str, Any], to_add: float
    ) -> float | bool:
        temp = 0.0
        key = poly_field
        if key not in binary_variables_name_weight:
            try:
                to_add *= float(key)
            except TypeError:
                print("Expression not supported\n")
                return False
        else:
            if isinstance(binary_variables_name_weight[key], list):
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1]
                        elif len(elem) == 3:
                            t = t * elem[1] + elem[2]
                        temp += t
            else:
                t = 1.0
                t *= binary_variables_name_weight[key][0]
                if len(binary_variables_name_weight[key]) == 2:
                    t *= binary_variables_name_weight[key][1]
                elif len(binary_variables_name_weight[key]) == 3:
                    t = t * binary_variables_name_weight[key][1] + binary_variables_name_weight[key][2]
                temp += t
            to_add *= temp
        return to_add

    def _convert_expression(self, expr: Expr, binary_variables_name_weight: dict[str, Any]) -> PUBO | bool:
        """function for translating an expression in the problem variable

        Keyword arguments:
        expr -- Expression to translate
        binary_variables_name_weight -- dictionary for the problem variables translation

        Return values:
        output_expr -- Translated expression
        """
        fields = str(expr).replace("**", "^").split(" ")
        func = PUBO()
        sign = "+"
        for field in fields:
            if field not in {"+", "-"}:
                poly_fields = field.split("*")
                to_add = 1.0
                for poly_field in poly_fields:
                    powers = poly_field.split("^")
                    if len(powers) == 2:
                        ret = self._convert_expression_power(powers, binary_variables_name_weight, to_add)
                    else:
                        ret = self._convert_expression_no_power(poly_field, binary_variables_name_weight, to_add)

                    if isinstance(ret, bool):
                        return False
                    to_add = ret

                if sign == "+":
                    func += to_add
                else:
                    func -= to_add
            else:
                sign = field
        return func

    @staticmethod
    def _convert_expression_prec_power(
        powers: list[str],
        binary_variables_name_weight: dict[str, Any],
        variables_dict: dict[str, Any],
        to_add: float,
        min_precision: float,
    ) -> tuple[float, float] | bool:
        temp = 0.0
        try:
            power = int(powers[1])
        except TypeError:
            print("Expression not supported\n")
            return False
        key = powers[0]
        if key not in binary_variables_name_weight:
            print("Expression not supported\n")
            return False
        if isinstance(binary_variables_name_weight[key], list):
            encoding = binary_variables_name_weight[key][0]
            if encoding == "dictionary":
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1] ** power
                        elif len(elem) == 3:
                            t = t * elem[1] ** power + elem[2] ** power
                        temp += t
            else:
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1]
                        elif len(elem) == 3:
                            t = t * elem[1] + elem[2]
                        temp += t
                temp **= power
        else:
            t = 1.0
            t *= binary_variables_name_weight[key][0]
            if len(binary_variables_name_weight[key]) == 2:
                t *= (binary_variables_name_weight[key][1]) ** power
            elif len(binary_variables_name_weight[key]) == 3:
                t = (t * binary_variables_name_weight[key][1] + binary_variables_name_weight[key][2]) ** power
            temp += t
        to_add *= temp
        name = key.split("_")
        if len(name) == 1:
            if variables_dict[name[0]].type == "c" and min_precision > variables_dict[name[0]].precision:
                min_precision = variables_dict[name[0]].precision
        elif (
            len(name) == 2
            and name[1].isnumeric()
            and name[0] in variables_dict
            and variables_dict[name[0]][int(name[1])].type == "c"
        ):
            min_precision = min(min_precision, variables_dict[name[0]][int(name[1])].precision)
        elif (
            len(name) == 3
            and name[1].isnumeric()
            and name[2].isnumeric()
            and name[0] in variables_dict
            and variables_dict[name[0]][int(name[1])][int(name[2])].type == "c"
        ):
            min_precision = min(min_precision, variables_dict[name[0]][int(name[1])][int(name[2])].precision)
        elif (
            len(name) == 4
            and name[1].isnumeric()
            and name[2].isnumeric()
            and name[3].isnumeric()
            and name[0] in variables_dict
            and variables_dict[name[0]][int(name[1])][int(name[2])][int(name[3])].type == "c"
            and min_precision > variables_dict[name[0]][int(name[1])][int(name[2])][int(name[3])].precision
        ):
            min_precision = variables_dict[name[0]][int(name[1])][int(name[2])][int(name[3])].precision
        elif (
            name is variables_dict.keys()
            and variables_dict[name[0]].type == "c"
            and min_precision > variables_dict[name[0]].precision
        ):
            min_precision = variables_dict[name[0]].precision

        return to_add, min_precision

    @staticmethod
    def _convert_expression_prec_no_power(
        poly_field: str,
        binary_variables_name_weight: dict[str, Any],
        variables_dict: dict[str, Any],
        to_add: float,
        min_precision: float,
    ) -> tuple[float, float] | bool:
        temp = 0.0
        key = poly_field
        if key not in binary_variables_name_weight:
            try:
                to_add *= float(key)
            except ValueError:
                print("Expression not supported\n")
                return False
        else:
            if isinstance(binary_variables_name_weight[key], list):
                for elem in binary_variables_name_weight[key]:
                    if not isinstance(elem, str):
                        t = 1.0
                        t *= elem[0]
                        if len(elem) == 2:
                            t *= elem[1]
                        elif len(elem) == 3:
                            t = t * elem[1] + elem[2]
                        temp += t
            else:
                t = 1.0
                t *= binary_variables_name_weight[key][0]
                if len(binary_variables_name_weight[key]) == 2:
                    t *= binary_variables_name_weight[key][1]
                elif len(binary_variables_name_weight[key]) == 3:
                    t = t * binary_variables_name_weight[key][1] + binary_variables_name_weight[key][2]
                temp += t
            to_add *= temp
            name = key.split("_")
            if len(name) == 1 and variables_dict[name[0]].type == "c":
                min_precision = min(min_precision, variables_dict[name[0]].precision)
            elif (
                len(name) == 2
                and name[1].isnumeric()
                and name[0] in variables_dict
                and variables_dict[name[0]][int(name[1])].type == "c"
                and min_precision > variables_dict[name[0]][int(name[1])].precision
            ):
                min_precision = variables_dict[name[0]][int(name[1])].precision
            elif len(name) == 3 and name[1].isnumeric() and name[2].isnumeric() and name[0] in variables_dict:
                if (
                    variables_dict[name[0]][int(name[1])][int(name[2])].type == "c"
                    and min_precision > variables_dict[name[0]][int(name[1])][int(name[2])].precision
                ):
                    min_precision = variables_dict[name[0]][int(name[1])][int(name[2])].precision
            elif (
                len(name) == 4
                and name[1].isnumeric()
                and name[2].isnumeric()
                and name[3].isnumeric()
                and name[0] in variables_dict
                and variables_dict[name[0]][int(name[1])][int(name[2])][int(name[3])].type == "c"
            ):
                min_precision = min(
                    min_precision,
                    variables_dict[name[0]][int(name[1])][int(name[2])][int(name[3])].precision,
                )
            elif (
                name is variables_dict.keys()
                and variables_dict[name[0]].type == "c"
                and min_precision > variables_dict[name[0]].precision
            ):
                min_precision = variables_dict[name[0]].precision
        return to_add, min_precision

    def _convert_expression_prec(
        self, expr: Expr, binary_variables_name_weight: dict[str, Any], variables_dict: dict[str, Any]
    ) -> tuple[PUBO, float] | bool:
        """function for translating an expression in the problem variable

        Keyword arguments:
        expr -- Expression to translate
        binary_variables_name_weight -- dictionary for the problem variables translation

        Return values:
        output_expr -- Translated expression
        """
        fields = str(expr).replace("**", "^").split(" ")
        func = PUBO()
        sign = "+"
        min_precision = 1.0
        for field in fields:
            if field not in {"+", "-"}:
                poly_fields = field.split("*")
                to_add = 1.0
                for poly_field in poly_fields:
                    powers = poly_field.split("^")

                    if len(powers) == 2:
                        ret = self._convert_expression_prec_power(
                            powers, binary_variables_name_weight, variables_dict, to_add, min_precision
                        )
                    else:
                        ret = self._convert_expression_prec_no_power(
                            poly_field, binary_variables_name_weight, variables_dict, to_add, min_precision
                        )
                    if isinstance(ret, bool):
                        return ret
                    to_add = ret[0]
                    min_precision = ret[1]

                if sign == "+":
                    func += to_add
                else:
                    func -= to_add
            else:
                sign = field
        return func, min_precision

    @staticmethod
    def _convert_expression_logic(expr: Expr, binary_variables_name_weight: dict[str, Any]) -> boolean_var | bool:
        """function for translating an expression in the problem variable in case of logic constraints

        Keyword arguments:
        expr -- Expression to translate
        binary_variables_name_weight -- dictionary for the problem variables translation

        Return values:
        Binary_var -- return the binary variable alone
        """
        key = str(expr)

        if key not in binary_variables_name_weight:
            print("Expression not supported\n")
            return False

        if isinstance(binary_variables_name_weight[key], list):
            print("Expression not supported\n")
            ret = False
        else:
            ret = binary_variables_name_weight[key][0]
        return ret

    @staticmethod
    def _min_max_const_estimation(exp: PUBO) -> tuple[float, float, float]:
        """function for estimating minimum and maximum value and the constraint value

        Keyword arguments:
        expr -- Expression to analyzed

        Return values:
        min_val -- float minimum value that the expression can assume
        max_val -- float maximum value that the expression can assume
        const -- float constant element of the expression
        """
        min_val = 0.0
        max_val = 0.0
        const = 0.0
        for key in exp:
            if len(key) == 0:
                const = exp[key]
            elif exp[key] > 0:
                max_val += exp[key]
            elif exp[key] < 0:
                min_val += exp[key]
        return min_val, max_val, const

    @staticmethod
    def _expression_to_hamiltonian_power(
        powers: list[str], binary_variables_name_weight_val: list[Any], to_add: float
    ) -> float | bool:
        try:
            power = int(powers[1])
        except TypeError:
            print("Expression not supported\n")
            return False
        key = powers[0]
        for elm in binary_variables_name_weight_val:
            if isinstance(elm, list):
                for el in elm:
                    if not isinstance(el, str) and key == (next(iter(el[0].variables))):
                        to_add *= el[0] ** power
                        break
            elif not isinstance(elm, str) and key == next(iter(elm[0].variables)):
                to_add *= elm[0] ** power
                break
        return to_add

    @staticmethod
    def _expression_to_hamiltonian_no_power(
        poly_field: str, binary_variables_name_weight_val: list[Any], to_add: float
    ) -> float:
        key = poly_field
        is_float = False
        try:
            temp = float(key)
        except ValueError:
            pass
        else:
            to_add *= temp
            is_float = True

        if not is_float:
            for elm in binary_variables_name_weight_val:
                if isinstance(elm, list):
                    for el in elm:
                        if not isinstance(el, str) and key == next(iter(el[0].variables)):
                            to_add *= el[0]
                elif not isinstance(elm, str) and key == next(iter(elm[0].variables)):
                    to_add *= elm[0]

        return to_add

    def _expression_to_hamiltonian(self, exp: Expr, binary_variables_name_weight_val: list[Any]) -> PUBO:
        """function for translating an expression in the problem variable into an Hamiltonian when is directly written with the inside binary variables

        Keyword arguments:
        expr -- Expression to translate
        binary_variables_name_weight_values -- list of the declared binary variables

        Return values:
        PUBO -- return corresponding PUBO expression
        """
        fields = str(exp).replace("**", "^").split(" ")
        hamiltonian = PUBO()
        sign = "+"
        for field in fields:
            if field not in {"+", "-"}:
                poly_fields = field.split("*")
                to_add = 1.0
                for poly_field in poly_fields:
                    powers = poly_field.split("^")
                    if len(powers) == 2:
                        t = self._expression_to_hamiltonian_power(powers, binary_variables_name_weight_val, to_add)
                        if not isinstance(t, bool):
                            to_add = t
                    else:
                        to_add = self._expression_to_hamiltonian_no_power(
                            poly_field, binary_variables_name_weight_val, to_add
                        )

                if sign == "+":
                    hamiltonian += to_add
                else:
                    hamiltonian -= to_add
            else:
                sign = field

        return hamiltonian
