"""Module for solving various NP-complete problems using quantum-inspired optimization techniques.

Contains problem definitions and solution methods for set cover, set packing, exact cover, hitting set, 3D-matching,
utilizing graph representations and solvers.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import networkx as nx
from sympy import Add, simplify

from mqt.qao.constraints import Constraints
from mqt.qao.objectivefunction import ObjectiveFunction
from mqt.qao.problem import Problem
from mqt.qao.solvers import Solver
from mqt.qao.variables import Variables

if TYPE_CHECKING:
    from collections.abc import Callable


class KarpSets:
    """Provides static methods to define and solve set-based problems such as set cover, set packing,.

    hitting set, exact cover, and 3D matching.

    Each method supports options for solving, visualizing, and outputting solutions to files.
    """

    @staticmethod
    def set_cover(
        input_data: str | list[tuple[int, list[int]]],
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        weighted: bool = False,
        solver_params: dict | None = None,
    ) -> Problem | list[tuple[int, list[int]]]:
        """Initializes and optionally solves a set cover problem.

        Args:
        input_data (str | list[tuple[int, list[int]]]): Input data as a filename or list of sets,
            where each set is represented by a tuple with its cost and list of elements.
        solve (bool): Specifies whether to solve the set cover problem (default is False).
        solver_method (Callable | None): An optional custom solver method to be used if solving.
        read_solution (Literal["print", "file"] | None): Defines how the solution should be outputted
            ("print" to display, "file" to save).
        weighted (bool): If True, uses weighted set cover formulation based on set costs (default is False).
        solver_params (dict | None): Additional parameters for the solver method, if provided.

        Returns:
        Problem | list[tuple[int, list[int]]]: Returns a Problem instance if solve is False;
        otherwise, returns a list of sets included in the solution.
        """
        b = 1.0
        a = b + b / 2

        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            file_name = input_data
            try:
                with Path.open(input_data) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_elements, num_sets = map(int, lines[0].strip().split())
            sets = []
            costs = []
            unique_elements = set()

            for line in lines[1:]:
                parts = list(map(int, line.strip().split()))
                cost = parts[0]
                costs.append(cost)
                elements = parts[2:]
                unique_elements.update(elements)
                sets.append((cost, elements))

        elif isinstance(input_data, list):
            file_name = ""
            sets = input_data
            num_sets = len(sets)
            unique_elements = set()
            costs = []
            for cost, elements in sets:
                costs.append(cost)
                unique_elements.update(elements)
            len(unique_elements)

        unique_elements = sorted(unique_elements)
        {elem: idx for idx, elem in enumerate(unique_elements)}
        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        set_vars = [variables.add_binary_variable(f"set_{i}") for i in range(num_sets)]

        if weighted:
            a = max(costs) * b + 1

        x_alpha_m_vars = {}
        for alpha in unique_elements:
            for m in range(1, num_sets + 1):
                x_alpha_m_vars[(alpha, m)] = variables.add_binary_variable(f"x_{alpha}_{m}")

        ha_terms = []

        for alpha in unique_elements:
            sum_x_alpha_m = Add(*[x_alpha_m_vars[(alpha, m)] for m in range(1, num_sets + 1)])
            term1 = a * (1 - sum_x_alpha_m) ** 2
            ha_terms.append(term1)

        for alpha in unique_elements:
            sum_m_x_alpha_m = Add(*[m * x_alpha_m_vars[(alpha, m)] for m in range(1, num_sets + 1)])
            sum_x_i_alpha = Add(*[set_vars[i] for i in range(num_sets) if alpha in sets[i][1]])
            term2 = a * (sum_m_x_alpha_m - sum_x_i_alpha) ** 2
            ha_terms.append(term2)

        ha = Add(*ha_terms)

        hb = b * Add(*[sets[i][0] * set_vars[i] for i in range(num_sets)]) if weighted else b * Add(*set_vars)

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

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("set_")}
        solution_sets = [
            (sets[int(var.split("_")[1])][0], sets[int(var.split("_")[1])][1])
            for var, value in set_variables.items()
            if value == 1.0
        ]

        if read_solution == "print":
            KarpSets.print_solution(
                "Set Cover Solution: ",
                file_name,
                KarpSets.set_to_string(solution_sets, weighted),
                KarpSets.convert_dict_to_string(KarpSets.check_set_cover_solution(input_data, solution_sets)),
            )
        elif read_solution == "file":
            KarpSets.save_solution(
                "Set Cover Solution: ",
                file_name,
                KarpSets.set_to_string(solution_sets, weighted),
                KarpSets.convert_dict_to_string(KarpSets.check_set_cover_solution(input_data, solution_sets)),
                input_data.replace(".txt", "") + "_set_cover_solution" + ".txt"
                if isinstance(input_data, str)
                else "set_cover_solution.txt",
            )

        return solution_sets

    @staticmethod
    def check_set_cover_solution(
        all_sets: list[tuple[int, list[int]]], solution: list[tuple[int, list[int]]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates a set cover solution by checking if all elements are covered.

        Args:
            all_sets (list[tuple[int, list[int]]]): All available sets, represented as tuples of
                costs and lists of elements.
            solution (list[tuple[int, list[int]]]): The proposed solution to check for coverage completeness.

        Returns:
            dict[str, bool | dict[str, list]]: Returns a dictionary with a validation status ("Valid Solution" key)
            and any errors, such as uncovered elements or missing sets.
        """
        errors = {"Missing Sets": [], "Uncovered Elements": []}

        universe = set()
        for _, elements in all_sets:
            universe.update(elements)

        covered = set()

        for s in solution:
            if s not in all_sets:
                errors["Missing Sets"].append(s)

        for _, elements in solution:
            covered.update(elements)

        if not covered >= universe:
            uncovered_elements = universe - covered
            errors["Uncovered Elements"] = list(uncovered_elements)

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def set_packing(
        input_data: str | list[tuple[int, list[int]]],
        t: int = 0,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
    ) -> Problem | list[tuple[int, list[int]]]:
        """Initializes and optionally solves a set packing problem.

        Args:
            input_data (str | list[tuple[int, list[int]]]): Input data as a filename or list of sets,
                where each set is represented by a tuple with its cost and list of elements.
            t (int): Optional constraint for the minimum number of sets required in the solution (default is 0).
            solve (bool): Specifies whether to solve the set packing problem (default is False).
            solver_method (Callable | None): An optional custom solver method to be used if solving.
            read_solution (Literal["print", "file"] | None): Defines how the solution should be outputted
                ("print" to display, "file" to save).
            solver_params (dict | None): Additional parameters for the solver method, if provided.

        Returns:
            Problem | list[tuple[int, list[int]]]: Returns a Problem instance if solve is False;
            otherwise, returns a list of sets in the solution.
        """
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        b = 1
        a = b + b / 2

        if isinstance(input_data, str):
            file_name = input_data
            try:
                with Path.open(input_data) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_elements, num_sets = map(int, lines[0].strip().split())
            sets = []

            for line in lines[1:]:
                parts = list(map(int, line.strip().split()))
                cost = parts[0]
                elements = parts[2:]
                sets.append((cost, elements))

        elif isinstance(input_data, list):
            file_name = ""
            sets = input_data
            num_sets = len(sets)

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        set_vars = [variables.add_binary_variable(f"set_{i}") for i in range(num_sets)]

        ha_terms = []

        for i in range(num_sets):
            for j in range(i + 1, num_sets):
                if set(sets[i][1]).intersection(sets[j][1]):
                    term = a * set_vars[i] * set_vars[j]
                    ha_terms.append(term)

        ha = simplify(Add(*ha_terms))

        hb = -b * Add(*set_vars)

        h = ha + hb

        objective_function.add_objective_function(h)

        if t != 0:
            constraint_expr = Add(*set_vars) - t
            constraints.add_constraint(str(constraint_expr) + ">= 0", True, True, True)

        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem
        solver = Solver()

        if solver_method is None:
            solver_method = solver.solve_simulated_annealing

        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("set_")}
        solution_sets = [
            (sets[int(var.split("_")[1])][0], sets[int(var.split("_")[1])][1])
            for var, value in set_variables.items()
            if value == 1.0
        ]

        if read_solution == "print":
            KarpSets.print_solution(
                "Set Packing Solution: ",
                file_name,
                KarpSets.set_to_string(solution_sets, False),
                KarpSets.convert_dict_to_string(KarpSets.check_set_packing_solution(input_data, solution_sets)),
            )
        elif read_solution == "file":
            KarpSets.save_solution(
                "Set Packing Solution: ",
                file_name,
                KarpSets.set_to_string(solution_sets, False),
                KarpSets.convert_dict_to_string(KarpSets.check_set_packing_solution(input_data, solution_sets)),
                txt_outputname=input_data.replace(".txt", "") + "_set_packing_solution" + ".txt"
                if isinstance(input_data, str)
                else "set_packing_solution.txt",
            )

        return solution_sets

    @staticmethod
    def check_set_packing_solution(
        all_sets: list[tuple[int, list[int]]], solution: list[tuple[int, list[int]]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates a set packing solution by ensuring no sets overlap.

        Args:
            all_sets (list[tuple[int, list[int]]]): All available sets, represented as tuples of
                costs and lists of elements.
            solution (list[tuple[int, list[int]]]): The proposed solution to validate.

        Returns:
            dict[str, bool | dict[str, list]]: Returns a dictionary with a validation status ("Valid Solution" key)
            and any overlapping sets.
        """
        errors = {"Missing Sets": [], "Overlapping Sets": []}

        used_elements = set()

        for s in solution:
            if s not in all_sets:
                errors["Missing Sets"].append(s)

        for cost, elements in solution:
            elements_set = set(elements)
            if any(elem in used_elements for elem in elements_set):
                errors["Overlapping Sets"].append((cost, elements))
            used_elements.update(elements_set)

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def hitting_set(
        input_data: str | list[tuple[int, list[int]]],
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
    ) -> Problem | list[int]:
        """Initializes and optionally solves a hitting set problem.

        Args:
            input_data (str | list[tuple[int, list[int]]]): Input data as a filename or list of sets,
                where each set is represented by a tuple with its cost and list of elements.
            solve (bool): Specifies whether to solve the hitting set problem (default is False).
            solver_method (Callable | None): An optional custom solver method to be used if solving.
            read_solution (Literal["print", "file"] | None): Defines how the solution should be outputted
                ("print" to display, "file" to save).
            solver_params (dict | None): Additional parameters for the solver method, if provided.

        Returns:
            Problem | list[int]: Returns a Problem instance if solve is False; otherwise, returns a list
            of elements in the hitting set solution.
        """
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        if isinstance(input_data, str):
            file_name = input_data
            try:
                with Path.open(input_data) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_elements, num_sets = map(int, lines[0].strip().split())
            sets = []
            costs = []
            sets_elements = []
            elements_sets = []
            unique_elements = set()

            for line in lines[1:]:
                parts = list(map(int, line.strip().split()))
                cost = parts[0]
                costs.append(cost)
                sets_elements.append(parts[2:])
                elements = parts[2:]
                unique_elements.update(elements)
                sets.append((cost, elements))

            for element in unique_elements:
                subsets = [i + 1 for i in range(num_sets) if element in sets_elements[i]]
                elements_sets.append(subsets)

            element_index_map = {}
            index_counter = 1

            for element in unique_elements:
                if element not in element_index_map.values():
                    element_index_map[index_counter] = element
                    index_counter += 1

            sets_list = [(1, list(elements)) for elements in elements_sets]

        elif isinstance(input_data, list):
            file_name = ""
            sets = input_data
            num_sets = len(sets)
            unique_elements = set()

            for _, elements in sets:
                unique_elements.update(elements)
            sets_list = []
            elements_sets = []

            for element in unique_elements:
                subsets = []
                for i in range(num_sets):
                    if element in sets[i][1]:
                        subsets.append(i + 1)
                elements_sets.append(subsets)

            element_index_map = {}
            index_counter = 1

            for element in unique_elements:
                if element not in element_index_map.values():
                    element_index_map[index_counter] = element
                    index_counter += 1

            sets_list = [(1, list(elements)) for elements in elements_sets]

        try:
            if not solve:
                return KarpSets.set_cover(sets_list, solve=False)

            problem = KarpSets.set_cover(sets_list, solve=False)
            solver = Solver()

            if solver_method is None:
                solver_method = solver.solve_simulated_annealing

            if solver_params is not None:
                solver_method = partial(solver_method, **solver_params)

            solution = solver_method(problem)
            solution_vars = solution.best_solution

            solution_set = [key for key, value in solution_vars.items() if key.startswith("set_") and value == 1.0]
            solution_set = [int(item.split("_")[1]) + 1 for item in solution_set]
            mapped_array = [element_index_map.get(item) for item in solution_set]

            if read_solution == "print":
                KarpSets.print_solution(
                    "Hitting Set Solution: ",
                    file_name,
                    f"H = {mapped_array}",
                    "Valid Solution"
                    if KarpSets.check_hitting_set_solution(input_data, solution_set)
                    else "Invalid Solution",
                )
            elif read_solution == "file":
                KarpSets.save_solution(
                    "Hitting Set Solution: ",
                    file_name,
                    f"H = {mapped_array}",
                    summary="Valid Solution"
                    if KarpSets.check_hitting_set_solution(input_data, solution_set)
                    else "Invalid Solution",
                    txt_outputname=input_data.replace(".txt", "") + "_hitting_set_solution" + ".txt"
                    if isinstance(input_data, str)
                    else "hitting_set_solution.txt",
                )

            return solution_set

        finally:
            pass

    @staticmethod
    def check_hitting_set_solution(
        sets: list[tuple[int, list[int]]], solution: list[int]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates a hitting set solution by ensuring that each set is "hit" by at least one element in the solution.

        Args:
        sets (list[tuple[int, list[int]]]): All available sets, represented as tuples of costs and lists of elements.
            Each tuple contains the cost of the set and a list of elements that belong to the set.
        solution (list[int]): The proposed solution list containing elements that form the hitting set.
            These elements should collectively "hit" each set by covering at least one element in each set.

        Returns:
        dict[str, bool | dict[str, list]]: A dictionary indicating the validation result. It includes:
            - "Valid Solution" (bool): True if the solution is valid, False otherwise.
            - "Errors" (dict): A dictionary of error details if the solution is invalid, containing:
                - "Invalid Solution Elements" (list[int]): Any elements in the solution that are not part of any set.
                - "Unhit Sets" (list[tuple[int, list[int]]]): Any sets that are not hit by any element in the solution.
        """
        solution_set = set(solution)

        all_elements = set()
        errors = {"Invalid Solution Elements": [], "Unhit Sets": []}

        for _, elements in sets:
            all_elements.update(elements)

        if not solution_set.issubset(all_elements):
            invalid_elements = solution_set - all_elements
            errors["Invalid Solution Elements"] = list(invalid_elements)

        for cost, elements in sets:
            if not any(elem in solution_set for elem in elements):
                errors["Unhit Sets"].append((cost, elements))

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def three_d_matching(
        input_data: str | nx.Graph,
        x: list[int],
        y: list[int],
        z: list[int],
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[tuple[int, int, int]]:
        """Initializes and optionally solves a 3D matching problem involving sets x, y, and z.

        Args:
            input_data (str | nx.Graph): Input data as a filename or a graph object defining the 3D matching problem.
            x (list[int]): The first set of elements for matching.
            y (list[int]): The second set of elements for matching.
            z (list[int]): The third set of elements for matching.
            solve (bool): Specifies whether to solve the 3D matching problem (default is False).
            solver_method (Callable | None): An optional custom solver method to be used if solving.
            read_solution (Literal["print", "file"] | None): Defines how the solution should be outputted
                ("print" to display, "file" to save).
            solver_params (dict | None): Additional parameters for the solver method, if provided.
            visualize (bool): If True, generates a visualization of the solution (default is False).

        Returns:
            Problem | list[tuple[int, int, int]]: Returns a Problem instance if solve is False;
            otherwise, returns a list of matching triples in the solution.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise TypeError(msg)

        if isinstance(input_data, str):
            try:
                file_name = input_data
                with Path.open(input_data) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_elements, num_edges = map(int, lines[0].strip().split())

            edges = []
            for line in lines[1:]:
                u, v = map(int, line.strip().split())
                edges.append((u, v))

        elif isinstance(input_data, nx.Graph):
            file_name = ""
            input_data.number_of_nodes()
            edges = list(input_data.edges())

        else:
            msg = "Invalid input type. Expected str or nx.Graph."
            raise TypeError(msg)

        all_triples = [(x1, y1, z1) for x1 in x for y1 in y for z1 in z]
        valid_triples = [
            triple for triple in all_triples if (triple[0], triple[1]) in edges and (triple[1], triple[2]) in edges
        ]

        set_packing_input = [(1, list(triple)) for triple in valid_triples]

        (f"_{file_name.replace('.txt', '')}", file_name) if file_name else ("", "")

        try:
            if not solve:
                return KarpSets.set_packing(set_packing_input, solve=False)

            problem = KarpSets.set_packing(set_packing_input, solve=False)
            solver = Solver()

            if solver_method is None:
                solver_method = solver.solve_simulated_annealing

            if solver_params is not None:
                solver_method = partial(solver_method, **solver_params)

            solution = solver_method(problem)
            solution_vars = solution.best_solution

            solution_set = [key for key, value in solution_vars.items() if key.startswith("set_") and value == 1.0]
            solution_set = [int(item.split("_")[1]) for item in solution_set]
            final_solution = [set_packing_input[i][1] for i in solution_set]

            formatted_strings = []

            for index, sub_array in enumerate(final_solution, start=1):
                tuple_representation = tuple(sub_array)
                formatted_strings.append(f"M{index} = {tuple_representation}")

            result_string = "\n".join(formatted_strings)

            if visualize and isinstance(input_data, nx.Graph):
                plt.figure()

                pos = {}
                for i, node in enumerate(x):
                    pos[node] = (0, -i)
                for i, node in enumerate(y):
                    pos[node] = (1, -i)
                for i, node in enumerate(z):
                    pos[node] = (2, -i)

                nx.draw(input_data, pos, with_labels=True, edge_color="black", node_size=700, node_color="lightblue")

                solution_edges = [(triple[0], triple[1]) for triple in final_solution] + [
                    (triple[1], triple[2]) for triple in final_solution
                ]
                nx.draw_networkx_edges(input_data, pos, edgelist=solution_edges, edge_color="blue", width=2)

                plt.show()

            if read_solution == "print":
                KarpSets.print_solution(
                    "3D Matching Solution: ",
                    file_name,
                    result_string,
                    KarpSets.convert_dict_to_string(KarpSets.check_three_d_matching(x, y, z, final_solution)),
                )
            elif read_solution == "file":
                KarpSets.save_solution(
                    "3D Matching Solution: ",
                    file_name,
                    result_string,
                    summary=KarpSets.convert_dict_to_string(KarpSets.check_three_d_matching(x, y, z, final_solution)),
                    txt_outputname=file_name.replace(".txt", "") + "_hitting_set_solution" + ".txt"
                    if isinstance(input_data, str)
                    else "hitting_set_solution.txt",
                )

            return final_solution

        finally:
            pass

    @staticmethod
    def check_three_d_matching(
        x: list[int], y: list[int], z: list[int], solution: list[tuple[int, int, int]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates a 3D matching solution by ensuring each element in the solution matches exactly one element.

        from each of the sets x, y, and z, and that there are no repeated or mismatched elements.

        Args:
        x (list[int]): The set of elements representing the first dimension of the matching.
        y (list[int]): The set of elements representing the second dimension of the matching.
        z (list[int]): The set of elements representing the third dimension of the matching.
        solution (list[tuple[int, int, int]]): A list of tuples, each representing a proposed match in
            the format (x_element, y_element, z_element). Each tuple should contain one element from each of x, y, and z.

        Returns:
        dict[str, bool | dict[str, list]]: A dictionary indicating the validation result. It includes:
            - "Valid Solution" (bool): True if the solution is valid, False otherwise.
            - "Errors" (dict): A dictionary of error details if the solution is invalid, containing:
                - "Invalid Triplets" (list[tuple[int, int, int]]): Any triplets that do not contain exactly one element from each of x, y, and z.
                - "Non-matching Elements" (list[tuple[int, int, int]]): Any triplets containing elements not found in x, y, or z.
                - "Repeated Elements" (list[tuple[int, int, int]]): Any elements that appear in more than one triplet, causing overlap.
                - "Unused Elements" (list[int]): Any elements in x, y, or z that do not appear in the solution.

        """
        used_elements = {}
        errors = {"Invalid Triplets": [], "Non-matching Elements": [], "Repeated Elements": [], "Unused Elements": []}

        for triplet in solution:
            if len(triplet) != 3:
                errors["Invalid Triplets"].append(triplet)
                continue

            x1, y1, z1 = triplet

            if x1 not in x or y1 not in y or z1 not in z:
                errors["Non-matching Elements"].append(triplet)
                continue

            repeated = False
            for element in [x1, y1, z1]:
                if element in used_elements:
                    errors["Repeated Elements"].append(used_elements[element])
                    errors["Repeated Elements"].append(triplet)
                    repeated = True
                    break

            if repeated:
                continue

            used_elements[x1] = triplet
            used_elements[y1] = triplet
            used_elements[z1] = triplet

        set(x + y + z)

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    # if A=1 and B=0 then it is an exact cover, if B=1 it is smallest exact cover, if B=-1 largest exact cover in number of subsets
    def exact_cover(
        input_data: str | list[tuple[int, list[int]]],
        b: float = 0,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
    ) -> Problem | list[tuple[int, list[int]]]:
        """Initializes and optionally solves an exact cover problem.

        Args:
            input_data (str | list[tuple[int, list[int]]]): Input data as a filename or list of sets,
                where each set is represented by a tuple with its cost and list of elements.
            b (float): Objective coefficient affecting the number of subsets selected; defaults to 0.
            solve (bool): Specifies whether to solve the exact cover problem (default is False).
            solver_method (Callable | None): An optional custom solver method to be used if solving.
            read_solution (Literal["print", "file"] | None): Defines how the solution should be outputted
                ("print" to display, "file" to save).
            solver_params (dict | None): Additional parameters for the solver method, if provided.

        Returns:
            Problem | list[tuple[int, list[int]]]: Returns a Problem instance if solve is False;
            otherwise, returns a list of sets forming the exact cover solution.
        """
        if any([solver_method is not None, read_solution is not None, solver_params is not None]) and not solve:
            msg = "'solve' must be True if 'solver_method', 'read_solution', or 'solver_params' are provided."
            raise ValueError(msg)

        a = 1.0

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(input_data) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_elements, num_sets = map(int, lines[0].strip().split())
            sets = []
            unique_elements = set()

            for line in lines[1:]:
                parts = list(map(int, line.strip().split()))
                cost = parts[0]
                elements = parts[2:]
                unique_elements.update(elements)
                sets.append((cost, elements))

        elif isinstance(input_data, list):
            filename = ""
            sets = input_data
            num_sets = len(sets)
            unique_elements = set()
            for _, elements in sets:
                unique_elements.update(elements)
            len(unique_elements)

        unique_elements = sorted(unique_elements)
        {elem: idx for idx, elem in enumerate(unique_elements)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        set_vars = [variables.add_binary_variable(f"set_{i}") for i in range(num_sets)]

        ha_terms = []

        for alpha in unique_elements:
            sum_x_i = Add(*[set_vars[i] for i in range(num_sets) if alpha in sets[i][1]])
            term = a * (sum_x_i - 1) ** 2
            ha_terms.append(term)

        ha = simplify(Add(*ha_terms))

        hb = b * Add(*set_vars)
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

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("set_")}

        solution_sets = [
            (sets[int(var.split("_")[1])][0], sets[int(var.split("_")[1])][1])
            for var, value in set_variables.items()
            if value == 1.0
        ]

        if read_solution == "print":
            KarpSets.print_solution(
                "Exact Cover Solution: ",
                filename,
                KarpSets.set_to_string(solution_sets, False),
                KarpSets.convert_dict_to_string(KarpSets.check_exact_cover_solution(input_data, solution_sets)),
            )
        elif read_solution == "file":
            KarpSets.save_solution(
                "Exact Cover Solution: ",
                filename,
                KarpSets.set_to_string(solution_sets, False),
                KarpSets.convert_dict_to_string(KarpSets.check_exact_cover_solution(input_data, solution_sets)),
                input_data.replace(".txt", "") + "_exact_cover_solution" + ".txt"
                if isinstance(input_data, str)
                else "exact_cover_solution.txt",
            )

        return solution_sets

    @staticmethod
    def check_exact_cover_solution(
        sets: list[tuple[int, list[int]]], solution: list[tuple[int, list[int]]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates an exact cover solution by ensuring each element is covered exactly once.

        and that there are no overlapping or missing elements.

        Args:
        sets (list[tuple[int, list[int]]]): A list of all available sets, where each set is represented
            by a tuple containing a cost and a list of elements within that set.
        solution (list[tuple[int, list[int]]]): The proposed exact cover solution as a list of sets,
            each represented by a tuple with a cost and a list of elements.

        Returns:
        dict[str, bool | dict[str, list]]: A dictionary indicating the validation result. It includes:
            - "Valid Solution" (bool): True if the solution is valid, False otherwise.
            - "Errors" (dict): A dictionary of error details if the solution is invalid, containing:
                - "Missing Sets" (list[tuple[int, list[int]]]): Any sets in the solution that do not match the original sets.
                - "Overlapping Sets" (list[tuple[int, list[int]]]): Any sets in the solution that share elements with other sets.
                - "Uncovered Elements" (list[int]): Any elements in the union of the original sets that are not covered by the solution.

        """
        all_elements = set()

        errors = {
            "Missing Sets": [],
            "Overlapping Sets": [],
            "Uncovered Elements": [],
        }

        for _, elements in sets:
            all_elements.update(elements)

        covered_elements = {}

        for cost, elements in solution:
            elements_tuple = tuple(elements)

            if (cost, elements_tuple) not in [(c, tuple(e)) for c, e in sets]:
                errors["Missing Sets"].append((cost, elements))

            overlapping_set = None
            for elem in elements:
                if elem in covered_elements:
                    overlapping_set = covered_elements[elem]

            if overlapping_set:
                errors["Overlapping Sets"].append(overlapping_set)
                errors["Overlapping Sets"].append((cost, elements))

            for elem in elements:
                covered_elements[elem] = (cost, elements)

        covered_set = set(covered_elements.keys())
        if covered_set != all_elements:
            uncovered = all_elements - covered_set
            errors["Uncovered Elements"] = list(uncovered)

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def set_to_string(sets_list: list[tuple[int, list[int]]], weighted: bool) -> str:
        """Converts a list of sets into a formatted string representation.

        Args:
            sets_list (list[tuple[int, list[int]]]): List of sets, each with a cost and elements.
            weighted (bool): If True, includes the cost of each set in the string representation.

        Returns:
            str: A formatted string representing each set in `sets_list`.
        """
        result = ""
        for idx, (cost, elements) in enumerate(sets_list, start=1):
            result += f"Set {idx} = {{{', '.join(map(str, elements))}}}"
            if weighted:
                result += f" with a cost of {cost}"
            result += "\n"
        return result.strip()

    @staticmethod
    def print_solution(
        problem_name: str | None = None,
        file_name: str | None = None,
        solution: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Prints the formatted solution of a problem to the console.

        Args:
            problem_name (str | None): The name of the problem being solved.
            file_name (str | None): The name of the file, if applicable.
            solution (str | None): The solution string to be printed.
            summary (str | None): An optional summary string with solution details.
        """
        start_str = problem_name + file_name
        print(start_str)
        print("=" * (len(start_str)))
        print(solution)
        print("-" * (len(start_str)))
        print(summary)

    @staticmethod
    def save_solution(
        problem_name: str | None = None,
        file_name: str | None = None,
        solution: str | None = None,
        summary: str | None = None,
        txt_outputname: str | None = None,
    ) -> None:
        """Saves the formatted solution of a problem to a specified file.

        Args:
            problem_name (str | None): The name of the problem being solved.
            file_name (str | None): The name of the file, if applicable.
            solution (str | None): The solution string to be saved.
            summary (str | None): An optional summary string with solution details.
            txt_outputname (str | None): The name of the output file.
        """
        start_str = problem_name + file_name
        with Path(txt_outputname).open("w") as f:
            f.write(start_str + "\n")
            f.write("=" * (len(start_str)) + "\n")
            f.write(solution + "\n")
            f.write("-" * (len(start_str)) + "\n")
            f.write(summary + "\n")
        print(f"Solution written to {txt_outputname}")

    @staticmethod
    def convert_dict_to_string(dictionary: dict) -> str:
        """Converts a dictionary of solution validation results into a readable string format.

        Args:
            dictionary (dict): A dictionary containing validation results and any error details.

        Returns:
            str: A formatted string representation of the validation results.
        """
        result = "Valid Solution" if dictionary.get("Valid Solution", False) else "Invalid Solution"

        for key, value in dictionary.items():
            if key != "Valid Solution" and isinstance(value, dict):
                result += f"\n{key}:"
                for sub_key, sub_value in value.items():
                    if sub_value:
                        result += f"\n'{sub_key}': {sub_value}"
        return result
