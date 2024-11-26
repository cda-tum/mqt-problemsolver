"""Module for solving various Karp's NP-complete problems using quantum-inspired optimization techniques.

Contains problem definitions and solution methods for SAT, 3-SAT, integer programming, knapsack, number partition,
and job sequencing, utilizing graph representations and solvers.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sympy import Add, simplify

from mqt.qao.constraints import Constraints
from mqt.qao.objectivefunction import ObjectiveFunction
from mqt.qao.problem import Problem
from mqt.qao.solvers import Solver
from mqt.qao.variables import Variables

from .karp_graphs import KarpGraphs

if TYPE_CHECKING:
    from collections.abc import Callable


class KarpNumber:
    """Provides static methods to define and solve a variety of NP-complete problems.

    Contains problem definitions and solution methods for SAT, 3-SAT, integer programming, knapsack, number partition, and job sequencing.
    Uses graph-based representations and solvers for quantum-inspired optimization.
    """

    @staticmethod
    def print_solution(
        problem_name: str = "",
        file_name: str = "",
        solution: str = "",
        summary: str = "",
    ) -> None:
        """Prints the formatted solution for a problem to the console."""
        start_str = problem_name + file_name
        print(start_str)
        print("=" * (len(start_str)))
        print(solution)
        print("-" * (len(start_str)))
        print(summary)

    @staticmethod
    def save_solution(
        problem_name: str = "",
        file_name: str = "",
        solution: str = "",
        summary: str = "",
        txt_outputname: str = "",
    ) -> None:
        """Saves the formatted solution to a specified file."""
        start_str = problem_name + file_name
        with Path(txt_outputname).open("w", encoding="utf-8") as f:
            f.write(start_str + "\n")
            f.write("=" * (len(start_str)) + "\n")
            f.write(solution + "\n")
            f.write("-" * (len(start_str)) + "\n")
            f.write(summary)

        print(f"Solution written to {txt_outputname}")

    @staticmethod
    def _create_graph(clauses: list[list[str]]) -> nx.Graph:
        """Creates a graph representation from a list of SAT clauses."""
        g = nx.Graph()

        for clause in clauses:
            for literal in clause:
                g.add_node(literal)

        for clause in clauses:
            for i in range(len(clause)):
                for j in range(i + 1, len(clause)):
                    g.add_edge(clause[i], clause[j])

        literals = {literal for clause in clauses for literal in clause}
        for literal in literals:
            complement = literal[1:] if literal.startswith("!") else "!" + literal
            if complement in literals:
                g.add_edge(literal, complement)

        return g

    @staticmethod
    def sat(
        input_data: str | list[tuple[int, int]],
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
    ) -> Problem | list[int] | dict[str, float]:
        """Initializes and optionally solves the SAT (Satisfiability) problem.

        Args:
        input_data (str | list[tuple[int, int]]): The SAT problem input, either as a filename or a list of clauses.
        solve (bool): Whether to solve the SAT problem; defaults to False.
        solver_method (Callable | None): Custom solver method to use if provided.
        read_solution (Literal["print", "file"] | None): Output method for solution, either 'print' or 'file'.
        solver_params (dict | None): Additional parameters for the solver method if provided.

        Returns:
        Problem | dict[str, float]: Returns a Problem instance if solve is False; otherwise, returns a dictionary of variable assignments.
        """
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            clauses = []
            for line in lines:
                newline = line.strip()
                if line:
                    literals = newline.split()
                    clauses.append(literals)
        elif all(isinstance(item, tuple) for item in input_data):
            # Convert list[tuple[int, int]] to list[list[str]]
            clauses = [[str(item[0]), str(item[1])] for item in input_data]
            filename = ""
        else:
            msg = "Invalid input_data type. Expected str or list[tuple[int, int]]."
            raise ValueError(msg)

        graph = KarpNumber._create_graph(clauses)

        problem = KarpGraphs.independent_set(graph, solve=False)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "SAT: "

        if isinstance(problem, Problem):
            solution = solver_method(problem)
        else:
            msg = "Expected `problem` to be of type `Problem`."
            raise TypeError(msg)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        output_dict = {
            k.split("_")[-1].replace("!", ""): (1.0 - v if "!" in k else v) for k, v in set_variables.items()
        }

        value_dict: dict[Any, Any] = {}
        for k, v in output_dict.items():
            value_dict.setdefault(v, []).append(k)

        result_list = []
        for value, keys in sorted(value_dict.items()):
            result_list.append(f"Value {int(value)} = {{{', '.join(sorted(keys))}}}")

        result_string = "\n".join(result_list)

        if read_solution == "print":
            KarpNumber.print_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(KarpNumber.check_three_sat_solution(input_data, output_dict)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_3sat_solution.txt"
                if isinstance(input_data, str)
                else "3sat_solution.txt"
            )
            KarpNumber.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(KarpNumber.check_three_sat_solution(input_data, output_dict)),
                txt_outputname=output_filename,
            )

        return output_dict

    @staticmethod
    def three_sat(
        input_data: str | list[tuple[int, int]],
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
    ) -> Problem | dict[str, float] | list[int]:
        """Initializes and optionally solves the 3-SAT (Satisfiability) problem.

        Args:
        input_data (str | list[tuple[int, int]]): The SAT problem input, either as a filename or a list of clauses.
        solve (bool): Whether to solve the SAT problem; defaults to False.
        solver_method (Callable | None): Custom solver method to use if provided.
        read_solution (Literal["print", "file"] | None): Output method for solution, either 'print' or 'file'.
        solver_params (dict | None): Additional parameters for the solver method if provided.

        Returns:
        Problem | dict[str, float]: Returns a Problem instance if solve is False; otherwise, returns a dictionary of variable assignments.
        """
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            clauses = []
            for line in lines:
                processed_line = line.strip()
                if processed_line:
                    literals = processed_line.split()
                    clauses.append(literals)

        elif all(isinstance(item, tuple) for item in input_data):
            # Convert list[tuple[int, int]] to list[list[str]]
            clauses = [[str(item[0]), str(item[1])] for item in input_data]
            filename = ""
        else:
            msg = "Invalid input_data type. Expected str or list[tuple[int, int]]."
            raise ValueError(msg)

        graph = KarpNumber._create_graph(clauses)

        problem = KarpGraphs.independent_set(graph, solve=False)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "3-SAT: "

        if isinstance(problem, Problem):
            solution = solver_method(problem)
        else:
            msg = "Expected `problem` to be of type `Problem`."
            raise TypeError(msg)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        output_dict = {
            k.split("_")[-1].replace("!", ""): (1.0 - v if "!" in k else v) for k, v in set_variables.items()
        }

        value_dict: dict[Any, Any] = {}
        for k, v in output_dict.items():
            value_dict.setdefault(v, []).append(k)

        result_list = []
        for value, keys in sorted(value_dict.items()):
            result_list.append(f"Value {int(value)} = {{{', '.join(sorted(keys))}}}")

        result_string = "\n".join(result_list)

        if read_solution == "print":
            KarpNumber.print_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(KarpNumber.check_three_sat_solution(clauses, output_dict)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_3sat_solution.txt"
                if isinstance(input_data, str)
                else "3sat_solution.txt"
            )
            KarpNumber.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(KarpNumber.check_three_sat_solution(clauses, output_dict)),
                txt_outputname=output_filename,
            )

        return output_dict

    @staticmethod
    def check_three_sat_solution(
        clauses: list[list[str]], solution: dict[str, float]
    ) -> dict[Any, Any]:
        """Validates a solution for the 3-SAT problem by checking clause satisfaction."""
        not_satisfied_clauses = []

        for clause in clauses:
            satisfied = False
            for literal in clause:
                var = literal.replace("!", "")
                is_negated = "!" in literal
                value = solution.get(var)

                if value is None:
                    return {
                        "Valid Solution": False,
                        "Solution": solution,
                        "Error": f"Variable {var} not found in the solution.",
                    }

                if (is_negated and value == 0.0) or (not is_negated and value == 1.0):
                    satisfied = True
                    break

            if not satisfied:
                not_satisfied_clauses.append(clause)

        if not_satisfied_clauses:
            return {"Valid Solution": False, "Not Satisfied Clauses": not_satisfied_clauses}

        return {"Valid Solution": True}

    @staticmethod
    def convert_dict_to_string(dictionary: dict[str, Any]) -> str:
        """Converts a dictionary of solution validation results into a readable string format."""
        result = "Valid Solution" if dictionary.get("Valid Solution", False) else "Invalid Solution"
        for key, value in dictionary.items():
            if key != "Valid Solution":
                result += f"\n{key}:"
                if isinstance(value, (list, tuple)):
                    for item in value:
                        result += f"\n{item}"
                else:
                    result += f" {value}"
        return result

    @staticmethod
    def integer_programming(
        input_data: str | list[list[int]],
        b: float = 1,
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
    ) -> Problem | list[float] | None:
        """Initializes and optionally solves an integer programming problem."""
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            num_variables, num_constraints = map(int, lines[0].strip().split())
            a = num_variables * b + 2 * b

            s = [list(map(int, lines[i].strip().split())) for i in range(1, num_constraints + 1)]

            b_vec = list(map(int, lines[num_constraints + 1].strip().split()))
            c_vec = list(map(int, lines[num_constraints + 2].strip().split()))

        elif isinstance(input_data, list):
            filename = ""
            data = input_data
            num_variables = len(data[0])
            num_constraints = len(data) - 2
            a = num_variables * b + 2 * b

            s = data[:num_constraints]
            b_vec = data[num_constraints]
            c_vec = data[num_constraints + 1]

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = [variables.add_binary_variable(f"x_{i + 1}") for i in range(num_variables)]

        ha_terms = []
        for j in range(num_constraints):
            sum_sx = Add(*[s[j][i] * x_vars[i] for i in range(num_variables)])
            ha_terms.append(a * (b_vec[j] - sum_sx) ** 2)

        ha = simplify(Add(*ha_terms))

        hb = -b * Add(*[c_vec[i] * x_vars[i] for i in range(num_variables)])

        h = ha + hb

        objective_function.add_objective_function(h)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)
        txt_outputname = "Integer Programming: "
        solution = solver_method(problem)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        if solution:
            set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
            values_array = [set_variables[f"x_{i + 1}"] for i in range(len(set_variables))]
            formatted_string = f"x = {values_array}"

            if read_solution == "print":
                KarpNumber.print_solution(
                    txt_outputname,
                    filename,
                    formatted_string,
                    KarpNumber.convert_dict_to_string(KarpNumber.check_integer_programming(s, b_vec, values_array)),
                )
            elif read_solution == "file":
                output_filename = (
                    filename.replace(".txt", "") + "_integer_programming.txt"
                    if isinstance(input_data, str)
                    else "integer_programming.txt"
                )
                KarpNumber.save_solution(
                    txt_outputname,
                    filename,
                    formatted_string,
                    KarpNumber.convert_dict_to_string(KarpNumber.check_integer_programming(s, b_vec, values_array)),
                    txt_outputname=output_filename,
                )

            return values_array

        print("No solution found")
        return None

    @staticmethod
    def check_integer_programming(s: list[list[int]], b: list[int], x: list[float]) -> dict[str, Any]:
        """Validates a solution for the integer programming problem by checking if constraints are met."""
        s_1 = np.array(s)
        b_1 = np.array(b)
        x_1 = np.array(x)

        result = np.dot(s_1, x_1)

        if np.array_equal(result, b_1):
            return {"Valid Solution": True}

        return {"Valid Solution": False, "Result": result}

    @staticmethod
    def knapsack(
        input_data: str | list[tuple[int, int]],
        max_weight: int,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
    ) -> Problem | list[tuple[int, int]]:
        """Initializes and optionally solves a knapsack problem given item weights and values."""
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            num_objects = len(lines)
            weights = []
            values = []
            for line in lines:
                weight, value = map(int, line.strip().split())
                weights.append(weight)
                values.append(value)

        elif isinstance(input_data, list):
            filename = ""
            data = input_data
            num_objects = len(data)
            weights = [item[0] for item in data]
            values = [item[1] for item in data]

        a = b * max(values) + b

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = [variables.add_binary_variable(f"x_{alpha + 1}") for alpha in range(num_objects)]

        y_vars = [variables.add_binary_variable(f"y_{i}") for i in range(1, max_weight + 1)]

        ha_terms = []
        sum_y_n = Add(*y_vars)
        ha_terms.append(a * (1 - sum_y_n) ** 2)

        nyi = Add(*[n * y_vars[n - 1] for n in range(1, max_weight + 1)])
        sum_wx = Add(*[weights[alpha] * x_vars[alpha] for alpha in range(num_objects)])
        ha_terms.append(a * (nyi - sum_wx) ** 2)

        ha = simplify(Add(*ha_terms))

        hb = -b * Add(*[values[alpha] * x_vars[alpha] for alpha in range(num_objects)])

        h = ha + hb

        objective_function.add_objective_function(h)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Knapsack: "
        solution = solver_method(problem)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}

        selected_items = [
            (weights[int(var.split("_")[1]) - 1], values[int(var.split("_")[1]) - 1])
            for var, value in set_variables.items()
            if value == 1.0
        ]

        formatted_strings = [f"Item {i + 1}: Weight = {w}, Value = {v}" for i, (w, v) in enumerate(selected_items)]
        result_string = "\n".join(formatted_strings)

        total_weight = sum(w for w, v in selected_items)
        total_value = sum(v for w, v in selected_items)

        summary_string = "Valid Solution" if total_weight <= max_weight else "Invalid Solution"
        summary_string += f"\nTotal Weight: {total_weight}, Total Value: {total_value}"

        if read_solution == "print":
            KarpNumber.print_solution(
                txt_outputname,
                filename,
                result_string,
                KarpNumber.convert_dict_to_string(KarpNumber.check_knapsack_solution(max_weight, selected_items)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_knapsack_solution.txt"
                if isinstance(input_data, str)
                else "knapsack_solution.txt"
            )
            KarpNumber.save_solution(
                txt_outputname, filename, result_string, summary_string, txt_outputname=output_filename
            )

        return selected_items

    @staticmethod
    def check_knapsack_solution(max_weight: int, selected_items: list[tuple[int, int]]) -> dict[str, bool | int]:
        """Validates a knapsack solution by checking weight limits and calculating total value."""
        total_weight = sum(w for w, v in selected_items)
        total_value = sum(v for w, v in selected_items)

        if total_weight <= max_weight:
            return {"Valid Solution": True, "Total Weight": total_weight, "Total Value": total_value}

        return {"Valid Solution": False, "Total Weight": total_weight, "Total Value": total_value}

    @staticmethod
    def number_partition(
        input_data: str | list[int],
        a: float = 1,
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
        visualize: bool = False,
    ) -> Problem | tuple[list[int], list[int]]:
        """Initializes and optionally solves a number partition problem to split elements into balanced subsets."""
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            num_elements = int(lines[0].strip())
            elements = [int(lines[i].strip()) for i in range(1, num_elements + 1)]

        elif isinstance(input_data, list):
            filename = ""
            elements = input_data
            num_elements = len(elements)

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        s_vars = [variables.add_spin_variable(f"s_{i + 1}") for i in range(num_elements)]

        sum_ns = Add(*[elements[i] * s_vars[i] for i in range(num_elements)])
        h_terms = [a * sum_ns**2]
        h = simplify(Add(*h_terms))

        objective_function.add_objective_function(h)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Number Partition: "
        solution = solver_method(problem)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("s_")}
        set_1 = []
        set_2 = []
        sum_1 = 0
        sum_2 = 0
        indices_set1 = []
        indices_set2 = []
        for var, value in set_variables.items():
            index = int(var.split("_")[1]) - 1
            if value == 1.0:
                indices_set1.append(index)
                sum_1 += elements[index]
                set_1.append(elements[index])
            else:
                indices_set2.append(index)
                sum_2 += elements[index]
                set_2.append(elements[index])

        if visualize:
            plt.bar(
                indices_set1,
                [elements[i] for i in indices_set1],
                color="blue",
                label="Set 1 (Sum:" + str(sum_1) + ")",
            )
            plt.bar(
                indices_set2,
                [elements[i] for i in indices_set2],
                color="red",
                label="Set 2 (Sum:" + str(sum_2) + ")",
            )
            plt.xlabel("Element Index")
            plt.ylabel("Element Value")
            plt.legend()
            plt.title("Number Partition Visualization")
            plt.show()

        formatted_strings: list[str] = []
        formatted_strings.extend((
            f"Set 1 = {{{','.join(map(str, set_1))}}}",
            f"Set 2 = {{{','.join(map(str, set_2))}}}",
        ))
        result_string = "\n".join(formatted_strings)

        if read_solution == "print":
            KarpNumber.print_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(
                    KarpNumber.check_number_partition_solution(elements, set_variables)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_number_partition_solution.txt"
                if isinstance(input_data, str)
                else "number_partition_solution.txt"
            )
            KarpNumber.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpNumber.convert_dict_to_string(
                    KarpNumber.check_number_partition_solution(elements, set_variables)
                ),
                txt_outputname=output_filename,
            )

        return set_1, set_2

    @staticmethod
    def check_number_partition_solution(
        elements: list[int], set_variables: dict[str, float]
    ) -> dict[str, int | bool | list[int]] | dict[Any, Any]:
        """Validates a number partition solution by comparing subset sums."""
        set_1 = []
        set_2 = []
        sum_1 = 0
        sum_2 = 0
        indices_set1 = []
        indices_set2 = []
        missing_elements = elements.copy()

        for var, value in set_variables.items():
            index = int(var.split("_")[1]) - 1
            if value == 1.0:
                indices_set1.append(index)
                sum_1 += elements[index]
                set_1.append(elements[index])
            else:
                indices_set2.append(index)
                sum_2 += elements[index]
                set_2.append(elements[index])
            missing_elements.remove(elements[index])

        result: dict[str, int | bool | list[int]] = {
            "Sum 1": sum_1,
            "Sum 2": sum_2,
        }

        if sum_1 == sum_2:
            result["Valid Solution"] = True
        else:
            result["Valid Solution"] = False

        if missing_elements:
            result["Missing Elements"] = missing_elements

        return result

    @staticmethod
    def job_sequencing(
        input_data: str | list[int],
        m: int,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable[..., Any] | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict[str, Any] | None = None,
    ) -> Problem | list[list[int]]:
        """Initializes and optionally solves a job sequencing problem to minimize scheduling conflicts. Pattern check for files is not included"""
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path(filename).open(encoding="utf-8") as file:
                    lines = file.readlines()
            except FileNotFoundError as err:
                print(f"Error: File {input_data} not found.")
                msg = f"Error: File {input_data} not found."
                raise FileNotFoundError(msg) from err

            num_jobs = int(lines[0].strip())

            job_lengths = []
            for i in range(1, num_jobs + 1):
                job_length = int(lines[i].strip())
                job_lengths.append(job_length)
            a = b * max(job_lengths) + b

        elif isinstance(input_data, list):
            filename = ""
            job_lengths = input_data
            num_jobs = len(job_lengths)
            a = b * max(job_lengths) + b

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {}
        for i in range(1, num_jobs + 1):
            for alpha in range(1, m + 1):
                x_vars[i, alpha] = variables.add_binary_variable(f"x_{i}_{alpha}")
        m_max = num_jobs * max(job_lengths)
        y_vars = {}
        for alpha in range(1, m + 1):
            for n in range(1, m_max + 1):
                y_vars[n, alpha] = variables.add_binary_variable(f"y_{n}_{alpha}")

        ha_terms = []
        for i in range(1, num_jobs + 1):
            sum_x_i = Add(*[x_vars[i, alpha] for alpha in range(1, m + 1)])
            ha_terms.append((1 - Add(sum_x_i)) ** 2)

        for alpha in range(1, m + 1):
            sum_n_y_n_alpha = Add(*[n * y_vars[n, alpha] for n in range(1, m_max + 1)])
            sum_i_n_alpha = Add(*[
                job_lengths[i - 1] * (x_vars[i, alpha] - x_vars[i, 1]) for i in range(1, num_jobs + 1)
            ])
            ha_terms.append((sum_n_y_n_alpha + sum_i_n_alpha) ** 2)

        ha = simplify(a * Add(*ha_terms))

        hb_terms = [b * job_lengths[i - 1] * x_vars[i, 1] for i in range(1, num_jobs + 1)]

        hb = simplify(Add(*hb_terms))

        h = ha + hb

        objective_function.add_objective_function(h)

        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Job Sequence " + str(m) + " clusters: "
        solution = solver_method(problem)

        if solution is None or not hasattr(solution, "best_solution"):
            msg = "Solver did not return a valid solution."
            raise ValueError(msg)

        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        result = defaultdict(list)

        for key, value in set_variables.items():
            if value == 1.0:
                sublist_index = int(key.split("_")[-1]) - 1
                result[sublist_index].append(int(key.split("_")[1]) - 1)

        max_index = max(result.keys()) if result else -1
        final_result = [result[i] for i in range(max_index + 1)]

        formatted_result = [
            f"Cluster {i + 1}: {{{', '.join(map(str, [job_lengths[job] for job in cluster]))}}}"
            for i, cluster in enumerate(final_result)
        ]
        result_string = "\n".join(formatted_result)

        if read_solution == "print":
            KarpNumber.print_solution(txt_outputname, filename, result_string, " ")
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_job_sequencing_solution.txt"
                if filename
                else "job_sequencing_solution.txt"
            )
            KarpNumber.save_solution(txt_outputname, filename, result_string, " ", txt_outputname=output_filename)

        return final_result
