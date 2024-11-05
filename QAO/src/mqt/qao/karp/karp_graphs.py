"""Module for solving various NP-complete problems using quantum-inspired optimization techniques.

Contains problem definitions and solution methods for clique, clique cover, vertex cover, graph colouring, feedback vertex set, feedback edge set,
hamiltonian path, travelling salesman path, utilizing graph representations and solvers.
"""

from __future__ import annotations

import operator
import random
from functools import partial
from itertools import combinations
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


class KarpGraphs:
    """A utility class for solving and validating NP-complete problems using quantum-inspired optimization.

    Supports problems like Clique, Vertex Cover, Graph Coloring, Hamiltonian Path, TSP, Independent Set,
    Max-Cut, and Directed Feedback Vertex/Edge Sets. Methods enable problem setup, solving, solution validation,
    and optional visualization.
    """

    @staticmethod
    def print_solution(
        problem_name: str | None = None,
        file_name: str | None = None,
        solution: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Prints the formatted solution with a header and separator.

        Args:
            problem_name (str | None): The name of the problem.
            file_name (str | None): Name of the file where the solution is saved.
            solution (str | None): Solution details to print.
            summary (str | None): Summary of the solution.
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
        """Saves the solution to a specified text file.

        Args:
            problem_name (str | None): The name of the problem.
            file_name (str | None): Name of the input file.
            solution (str | None): Solution details to save.
            summary (str | None): Summary of the solution.
            txt_outputname (str | None): Output file name to save solution.
        """
        start_str = problem_name + file_name
        with Path.open(txt_outputname, "w") as f:
            f.write(start_str + "\n")
            f.write("=" * (len(start_str)) + "\n")
            f.write(solution + "\n")
            f.write("-" * (len(start_str)) + "\n")
            f.write(summary + "\n")
        print(f"Solution written to {txt_outputname}")

    @staticmethod
    def clique_cover(
        input_data: str | nx.Graph,
        num_colors: int,
        a: float = 1,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        txt_outputname: str | None = None,
        visualize: bool = False,
    ) -> Problem | list[tuple[int, int]]:
        """Solves the clique cover problem for a given graph and number of colors.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            num_colors (int): Number of colors.
            a (float): Penalty parameter for constraints. Default is 1.
            b (float): Penalty parameter for constraints. Default is 1.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            txt_outputname (str | None): File name for text output.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        degree = dict.fromkeys(unique_vertices, 0)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        max_degree = max(degree.values())
        a = (max_degree + 2) * b

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {
            (v, i): variables.add_binary_variable(f"x_{v}_{i}")
            for v in unique_vertices
            for i in range(1, num_colors + 1)
        }

        ha_terms = []

        for v in unique_vertices:
            sum_x_v = Add(*[x_vars[v, i] for i in range(1, num_colors + 1)])
            ha_terms.append(a * (1 - sum_x_v) ** 2)

        hb_terms = []
        for i in range(1, num_colors + 1):
            sum_edges_1 = 0.5 * (-1 + Add(*[x_vars[v, i] for v in unique_vertices]))
            sum_edges_2 = Add(*[x_vars[v, i] for v in unique_vertices])
            sum_edges_exist = Add(*[x_vars[u, i] * x_vars[v, i] for u, v in edges])
            hb_terms.append(b * (sum_edges_1 * sum_edges_2 - sum_edges_exist))

        ha = simplify(Add(*ha_terms))
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

        txt_outputname = f"Clique-cover with {num_colors} colors: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}

        solution_output = []
        for var, value in set_variables.items():
            if value == 1.0:
                parts = var.split("_")
                solution_output.append((int(parts[1]), int(parts[2])))

        result_string = "\n".join([f"Vertex:{v} Color:{c}" for v, c in solution_output])

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            distinct_colors = {color_index for _, color_index in solution_output}
            max_color_index = max(distinct_colors)

            random_colors = {}
            for color_index in range(1, max_color_index + 1):
                random_colors[color_index] = random.choice([
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "purple",
                    "cyan",
                    "magenta",
                    "orange",
                    "brown",
                    "pink",
                    "lime",
                    "teal",
                    "lavender",
                    "maroon",
                    "navy",
                ])

            node_colors = []
            for vertex_index in unique_vertices:
                assigned_color = None
                for node, color_index in solution_output:
                    if node == vertex_index:
                        assigned_color = random_colors[color_index]
                        break
                node_colors.append(assigned_color)

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title(f"Clique Cover with {num_colors} colors")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                result_string,
                KarpGraphs.convert_dict_to_string(KarpGraphs.check_clique_cover_solution(input_data, solution_output)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_clique_cover_solution.txt"
                if isinstance(input_data, str)
                else "clique_cover_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_clique_cover_solution(input_data, solution_output)
                ),
                txt_outputname=output_filename,
            )

        return solution_output

    @staticmethod
    def check_clique_cover_solution(
        graph: nx.Graph, cover: list[tuple[int, int]]
    ) -> dict[str, bool | dict[str, list | dict[int, list[tuple[int, int]]]]]:
        """Validates the solution for the clique cover problem.

        Args:
            graph (nx.Graph): Input graph.
            cover (list[tuple[int, int]]): Solution with vertices and assigned colors.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Missing Nodes": [], "Invalid Cliques": {}}

        color_dict = dict(cover)

        for node in graph.nodes:
            if node not in color_dict:
                errors["Missing Nodes"].append(node)

        cliques = {}
        for node, color in color_dict.items():
            if color not in cliques:
                cliques[color] = set()
            cliques[color].add(node)

        for color, clique in cliques.items():
            invalid_pairs = [
                (node1, node2)
                for node1 in clique
                for node2 in clique
                if node1 != node2 and not graph.has_edge(node1, node2)
            ]
            if invalid_pairs:
                errors["Invalid Cliques"][color] = invalid_pairs

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def graph_coloring(
        input_data: str | nx.Graph,
        num_colors: int,
        a: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        txt_outputname: str | None = None,
        visualize: bool = False,
    ) -> Problem | list[tuple[int, int]]:
        """Solves the graph coloring problem for a given graph and number of colors.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            num_colors (int): Number of colors.
            a (float): Penalty parameter for constraints. Default is 1.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            txt_outputname (str | None): File name for text output.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {
            (v, i): variables.add_binary_variable(f"x_{v}_{i}")
            for v in unique_vertices
            for i in range(1, num_colors + 1)
        }

        ha_terms = []

        for v in unique_vertices:
            sum_x_v_i = Add(*[x_vars[v, i] for i in range(1, num_colors + 1)])
            term1 = a * (1 - sum_x_v_i) ** 2
            ha_terms.append(term1)

        for u, v in edges:
            for i in range(1, num_colors + 1):
                term2 = a * x_vars[u, i] * x_vars[v, i]
                ha_terms.append(term2)

        ha = simplify(Add(*ha_terms))

        objective_function.add_objective_function(ha)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem
        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = f"{num_colors}-colored Graph: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}

        solution_output = []
        for var, value in set_variables.items():
            if value == 1.0:
                parts = var.split("_")
                solution_output.append((int(parts[1]), int(parts[2])))

        result_string = "\n".join([f"Vertex:{v} Color:{c}" for v, c in solution_output])

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            distinct_colors = {color_index for _, color_index in solution_output}
            max_color_index = max(distinct_colors)

            available_colors = [
                "red",
                "blue",
                "green",
                "yellow",
                "purple",
                "cyan",
                "magenta",
                "orange",
                "brown",
                "pink",
                "lime",
                "teal",
                "lavender",
                "maroon",
                "navy",
            ]

            random_colors = {}
            for color_index in range(1, max_color_index + 1):
                chosen_color = random.choice(available_colors)
                random_colors[color_index] = chosen_color
                available_colors.remove(chosen_color)

            node_colors = []
            for vertex_index in unique_vertices:
                assigned_color = None
                for node, color_index in solution_output:
                    if node == vertex_index:
                        assigned_color = random_colors[color_index]
                        break
                node_colors.append(assigned_color)

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title(f"Graph Coloring with {num_colors} colors")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                result_string,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_graph_coloring_solution(input_data, solution_output)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_graph_coloring_solution.txt"
                if isinstance(input_data, str)
                else "graph_coloring_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_graph_coloring_solution(input_data, solution_output)
                ),
                txt_outputname=output_filename,
            )

        return solution_output

    @staticmethod
    def check_graph_coloring_solution(
        graph: nx.Graph, coloring: list[tuple[int, int]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates the solution for the graph coloring problem.

        Args:
            graph (nx.Graph): Input graph.
            coloring (list[tuple[int, int]]): Solution with vertices and assigned colors.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Missing Nodes": [], "Conflicting Edges": []}

        color_dict = dict(coloring)

        for node in graph.nodes:
            if node not in color_dict:
                errors["Missing Nodes"].append(node)

        for node1, node2 in graph.edges:
            if color_dict.get(node1) == color_dict.get(node2):
                errors["Conflicting Edges"].append((node1, node2))

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def vertex_cover(
        input_data: str | nx.Graph,
        a: float = 1,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        txt_outputname: str | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the vertex cover problem for a given graph.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            a (float): Penalty parameter for constraints. Default is 1.
            b (float): Penalty parameter for constraints. Default is 1.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            txt_outputname (str | None): File name for text output.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        a = b + 0.5

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {v: variables.add_binary_variable(f"x_{v}") for v in unique_vertices}

        ha_terms = []

        for u, v in edges:
            term1 = a * (1 - x_vars[u]) * (1 - x_vars[v])
            ha_terms.append(term1)

        ha = simplify(Add(*ha_terms))

        hb = b * Add(*[x_vars[v] for v in unique_vertices])

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

        txt_outputname = "Vertex Cover: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        output_l = "C = {"
        solution_output = [int(var[len("x_") :]) for var, value in set_variables.items() if value == 1.0]
        output_l += ", ".join(map(str, solution_output)) + "}"

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            node_colors = ["lightgreen" if v in solution_output else "grey" for v in unique_vertices]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title("Vertex Cover")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_l,
                KarpGraphs.convert_dict_to_string(KarpGraphs.check_vertex_cover_solution(input_data, solution_output)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_vertex_cover_solution.txt"
                if isinstance(input_data, str)
                else "vertex_cover_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_l,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_vertex_cover_solution(input_data, solution_output)
                ),
                txt_outputname=output_filename,
            )

        return solution_output

    @staticmethod
    def check_vertex_cover_solution(
        graph: nx.Graph, solution: list[int]
    ) -> dict[str, bool | dict[str, list[tuple[int, int]]]]:
        """Validates the solution for the vertex cover problem.

        Args:
            graph (nx.Graph): Input graph.
            solution (list[int]): Solution vertices in the cover.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Uncovered Edges": []}

        cover_set = set(solution)

        for edge in graph.edges:
            node1, node2 = edge
            if node1 not in cover_set and node2 not in cover_set:
                errors["Uncovered Edges"].append(edge)

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def clique(
        input_data: str | nx.Graph,
        k: int = 0,
        a: float = 1,
        b: float = 2,
        c: float = 2,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        txt_outputname: str | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the clique problem for a given graph and size of the clique.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            k (int): Desired size of the clique, or 0 for the largest clique.
            a (float): Penalty parameter for constraints. Default is 1.
            b (float): Penalty parameter for constraints. Default is 2.
            c (float): Penalty parameter for constraints. Default is 2.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            txt_outputname (str | None): File name for text output.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {v: variables.add_binary_variable(f"x_{v}") for v in unique_vertices}

        if k == 0:
            degree = dict.fromkeys(unique_vertices, 0)
            for u, v in edges:
                degree[u] += 1
                degree[v] += 1
            max_degree = max(degree.values())
            b = c
            a = (max_degree + 2) * b
            y_vars = {i: variables.add_binary_variable(f"y_{i}") for i in range(2, max_degree + 1)}

            ha_terms = []
            ha_terms.extend((
                a * (1 - Add(*[y_vars[i] for i in range(2, max_degree + 1)])) ** 2,
                a
                * (Add(*[i * y_vars[i] for i in range(2, max_degree + 1)]) - Add(*[x_vars[p] for p in unique_vertices]))
                ** 2,
            ))

            ha = simplify(Add(*ha_terms))

            hb_terms = []
            nyi = (
                b
                * 0.5
                * Add(*[i * y_vars[i] for i in range(2, max_degree + 1)])
                * (-1 + Add(*[i * y_vars[i] for i in range(2, max_degree + 1)]))
            )
            sum_x_u_x_v = b * -1 * Add(*[x_vars[u] * x_vars[v] for u, v in edges])
            hb_terms.extend((nyi, sum_x_u_x_v))

            hb = simplify(Add(*hb_terms))

            hc = -c * Add(*[x_vars[v] for v in unique_vertices])

            h = ha + hb + hc
        else:
            a = k * b + 1

            ha = a * (k - Add(*[x_vars[v] for v in unique_vertices])) ** 2

            hb = b * (k * (k - 1) / 2 - Add(*[x_vars[u] * x_vars[v] for u, v in edges]))

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
        txt_outputname = "Largest Clique: " if k == 0 else str(k) + "-sized Clique: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        output_l = "Clique = {"
        solution_output = [int(var[len("x_") :]) for var, value in set_variables.items() if value == 1.0]
        output_l += ", ".join(map(str, solution_output)) + "}"

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            node_colors = ["lightgreen" if v in solution_output else "grey" for v in unique_vertices]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title("Clique")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_l,
                KarpGraphs.convert_dict_to_string(KarpGraphs.check_clique_solution(input_data, solution_output)),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_clique_solution.txt"
                if isinstance(input_data, str)
                else "clique_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_l,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_vertex_cover_solution(input_data, solution_output)
                ),
                txt_outputname=output_filename,
            )

        return solution_output

    @staticmethod
    def check_clique_solution(
        graph: nx.Graph, solution: list[int]
    ) -> dict[str, bool | dict[str, list[tuple[int, int]]]]:
        """Validates a solution for the clique problem, ensuring all nodes in the solution form a complete subgraph.

        Args:
            graph (nx.Graph): The input graph.
            solution (list[int]): List of nodes in the proposed clique.

        Returns:
            dict: A dictionary with "Valid Solution" status (True/False) and details of any missing edges
                  that prevent the solution from being a valid clique.
        """
        errors = {"Non-Clique Pairs": []}

        for node1, node2 in combinations(solution, 2):
            if not graph.has_edge(node1, node2):
                errors["Non-Clique Pairs"].append((node1, node2))

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def hamiltonian_path(
        input_data: str | nx.Graph,
        cycle: bool = False,
        a: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the Hamiltonian path problem for a given graph.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            cycle (bool): Whether to find a Hamiltonian cycle instead of a path. Default is False.
            a (float): Penalty parameter for constraints. Default is 1.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        # CYCLE = TRUE DOESBT WORK
        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = set()
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.add((u, v))
                edges.add((v, u))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            num_vertices = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = set(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)  # Sort to maintain a consistent order
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {
            (v, j): variables.add_binary_variable(f"x_{v}_{j}")
            for v in unique_vertices
            for j in range(1, num_vertices + 1)
        }

        h_terms = []

        for v in unique_vertices:
            sum_x_v_j = Add(*[x_vars[v, j] for j in range(1, num_vertices + 1)])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        for j in range(1, num_vertices + 1):
            sum_x_v_j = Add(*[x_vars[v, j] for v in unique_vertices])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        range_j = num_vertices + 1 if cycle else num_vertices

        for j in range(1, range_j):
            for u in unique_vertices:
                for v in unique_vertices:
                    if u != v and (u, v) not in edges:
                        if j == num_vertices:
                            h_terms.append(a * x_vars[u, j])
                            continue
                        h_terms.append(a * x_vars[u, j] * x_vars[v, j + 1])

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
        txt_outputname = "Hamiltonian Cycle: " if cycle else "Hamiltonian Path:"

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_") and v == 1.0}

        parsed_pairs = [(int(key.split("_")[1]), int(key.split("_")[2])) for key in set_variables]
        sorted_pairs = sorted(parsed_pairs, key=operator.itemgetter(1))
        sorted_vertices = [vertex for vertex, time_step in sorted_pairs]
        output_path = "Path = " + ",".join(map(str, sorted_vertices))
        if cycle:
            output_path += "," + str(sorted_vertices[0])

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            path_edges = [(sorted_vertices[i], sorted_vertices[i + 1]) for i in range(len(sorted_vertices) - 1)]
            if cycle:
                path_edges.append((sorted_vertices[-1], sorted_vertices[0]))

            edge_colors = [
                "blue" if (u, v) in path_edges or (v, u) in path_edges else "black" for u, v in graph.edges()
            ]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color="lightblue", edge_color=edge_colors, with_labels=True)

            plt.title("Hamiltonian Path")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_path,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_hamiltonian_path_solution(input_data, sorted_vertices)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_hamiltonian_solution.txt"
                if isinstance(input_data, str)
                else "hamiltonian_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_path,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_hamiltonian_path_solution(input_data, sorted_vertices)
                ),
                txt_outputname=output_filename,
            )

        return sorted_vertices

    @staticmethod
    def check_hamiltonian_path_solution(graph: nx.Graph, solution: list[int]) -> dict[str, bool | dict[str, list]]:
        """Validates the solution for the Hamiltonian path problem.

        Args:
            graph (nx.Graph): Input graph.
            solution (list[int]): Solution vertices in the path.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Missing Nodes": [], "Invalid Edges": []}

        if set(solution) != set(graph.nodes):
            missing_nodes = set(graph.nodes) - set(solution)
            errors["Missing Nodes"] = list(missing_nodes)

        for i in range(len(solution) - 1):
            if not graph.has_edge(solution[i], solution[i + 1]):
                errors["Invalid Edges"].append((solution[i], solution[i + 1]))

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def _hamiltonian_path_tsp(filename: str, cycle: bool = False, a: float = 1, b: float = 1) -> tuple:
        """Sets up the Hamiltonian path or Travelling Salesman Problem (TSP) as an optimization problem.

        Args:
            filename (str): Path to the file containing graph data.
            cycle (bool): If True, constructs a Hamiltonian cycle or TSP cycle instead of a path. Default is False.
            a (float): Penalty parameter to enforce constraints. Default is 1.
            b (float): Weight multiplier for edge distances in TSP. Default is 1.

        Returns:
            tuple: Contains the problem Hamiltonian, variable definitions, weighted edges, and binary variables
                   for each vertex and position in the path.
        """
        try:
            with Path.open(filename) as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return None, None, None, None

        num_vertices, num_edges = map(int, lines[0].strip().split())
        edges_w = []
        unique_vertices = set()

        for i in range(1, num_edges + 1):
            u, v, weight = map(int, lines[i].strip().split())
            edges_w.extend(((u, v, weight), (v, u, weight)))
            unique_vertices.update([u, v])

        weights = [weight for _, _, weight in edges_w]
        max_weight = max(weights)
        a = max_weight * b + 1
        edges = set()

        for i in range(1, num_edges + 1):
            u, v, _ = map(int, lines[i].strip().split())
            edges.add((u, v))
            edges.add((v, u))

        unique_vertices = sorted(unique_vertices)

        variables = Variables()

        x_vars = {
            (v, j): variables.add_binary_variable(f"x_{v}_{j}")
            for v in unique_vertices
            for j in range(1, num_vertices + 1)
        }

        h_terms = []

        for v in unique_vertices:
            sum_x_v_j = Add(*[x_vars[v, j] for j in range(1, num_vertices + 1)])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        for j in range(1, num_vertices + 1):
            sum_x_v_j = Add(*[x_vars[v, j] for v in unique_vertices])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        range_j = num_vertices + 1 if cycle else num_vertices

        for j in range(1, range_j):
            for u in unique_vertices:
                for v in unique_vertices:
                    if u != v and (u, v) not in edges:
                        if j == num_vertices:
                            h_terms.append(a * x_vars[u, j])
                            continue
                        h_terms.append(a * x_vars[u, j] * x_vars[v, j + 1])

        ha = simplify(Add(*h_terms))

        return ha, variables, edges_w, x_vars

    @staticmethod
    def travelling_salesman(
        input_data: str | nx.Graph,
        a: float = 1,
        b: float = 1,
        cycle: bool = False,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the Travelling Salesman Problem (TSP) or Hamiltonian path problem for a graph.

        Args:
            input_data (str | nx.Graph): Graph data as a file path or a NetworkX graph.
            a (float): Penalty parameter for constraints. Default is 1.
            b (float): Weight multiplier for edge distances. Default is 1.
            cycle (bool): If True, solves for a TSP cycle; otherwise, solves for a path. Default is False.
            solve (bool): If True, solves the problem; if False, returns the problem setup. Default is False.
            solver_method (Callable | None): Optional solver function.
            read_solution (Literal["print", "file"] | None): Specifies output format.
            solver_params (dict | None): Additional solver parameters.
            visualize (bool): If True, displays the solution graph. Default is False.

        Returns:
            Problem | list[int]: The formulated problem or, if solved, a list of vertices in the solution path.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {filename} not found.")
                return None, None, None, None

            num_vertices, num_edges = map(int, lines[0].strip().split())
            edges_w = []
            unique_vertices = set()
            edges = set()

            for i in range(1, num_edges + 1):
                u, v, weight = map(int, lines[i].strip().split())
                edges_w.extend(((u, v, weight), (v, u, weight)))
                unique_vertices.update([u, v])
                edges.add((u, v))
                edges.add((v, u))

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            num_vertices = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges_w = [(u, v, graph[u][v]["weight"]) for u, v in graph.edges()]
            unique_vertices = set(graph.nodes())
            edges = set()
            for u, v in graph.edges():
                edges.add((u, v))
                edges.add((v, u))

        weights = [weight for _, _, weight in edges_w]
        max_weight = max(weights)
        a = max_weight * b + 1

        unique_vertices = sorted(unique_vertices)

        variables = Variables()

        x_vars = {
            (v, j): variables.add_binary_variable(f"x_{v}_{j}")
            for v in unique_vertices
            for j in range(1, num_vertices + 1)
        }

        h_terms = []

        for v in unique_vertices:
            sum_x_v_j = Add(*[x_vars[v, j] for j in range(1, num_vertices + 1)])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        for j in range(1, num_vertices + 1):
            sum_x_v_j = Add(*[x_vars[v, j] for v in unique_vertices])
            h_terms.append(a * (1 - sum_x_v_j) ** 2)

        range_j = num_vertices + 1 if cycle else num_vertices

        for j in range(1, range_j):
            for u in unique_vertices:
                for v in unique_vertices:
                    if u != v and (u, v) not in edges:
                        if j == num_vertices:
                            h_terms.append(a * x_vars[u, j])
                            continue
                        h_terms.append(a * x_vars[u, j] * x_vars[v, j + 1])

        ha = simplify(Add(*h_terms))

        unique_vertices = {v for edge in edges for v in edge[:2]}
        num_vertices = len(unique_vertices)

        hb_terms = []

        hb_terms = [
            b * weight * x_vars[u, j] * x_vars[v, j + 1] for u, v, weight in edges_w for j in range(1, num_vertices)
        ]

        hb = simplify(Add(*hb_terms))

        h = ha + hb

        objective_function = ObjectiveFunction()
        objective_function.add_objective_function(h)

        constraints = Constraints()

        problem = Problem()
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)
        txt_outputname = "Travelling-Salesman Cycle: " if cycle else "Travelling-Salesman Path:"

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_") and v == 1.0}
        edges_dict = {(u, v): weight for u, v, weight in edges_w}
        parsed_pairs = [(int(key.split("_")[1]), int(key.split("_")[2])) for key in set_variables]
        sorted_pairs = sorted(parsed_pairs, key=operator.itemgetter(1))
        sorted_vertices = [vertex for vertex, time_step in sorted_pairs]
        print(sorted_vertices)
        total_cost = 0
        formatted_result = []
        for i in range(len(sorted_vertices) - 1):
            u, v = sorted_vertices[i], sorted_vertices[i + 1]
            cost = edges_dict.get((u, v)) if (u, v) in edges_dict else edges_dict.get((v, u))
            total_cost += cost
            formatted_result.append(f"Step {i + 1}. from {u} to {v} with cost of {cost}")
        if cycle:
            u, v = sorted_vertices[-1], sorted_vertices[0]
            cost = edges_dict.get((u, v)) if (u, v) in edges_dict else edges_dict.get((v, u))
            total_cost += cost
            formatted_result.append(f"Step {len(sorted_vertices)}. from {u} to {v} with cost of {cost}")

        result_string = "\n".join(formatted_result)

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_weighted_edges_from(edges_w)

            path_edges = [(sorted_vertices[i], sorted_vertices[i + 1]) for i in range(len(sorted_vertices) - 1)]
            if cycle:
                path_edges.append((sorted_vertices[-1], sorted_vertices[0]))

            edge_colors = [
                "blue" if (u, v) in path_edges or (v, u) in path_edges else "black" for u, v in graph.edges()
            ]

            pos = nx.spring_layout(graph)
            nx.draw(
                graph,
                pos,
                node_color="lightblue",
                edge_color=edge_colors,
                with_labels=True,
                node_size=500,
                font_size=10,
            )

            labels = {(u, v): str(graph[u][v]["weight"]) for u, v in graph.edges()}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color="black")

            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="blue", width=2.0)
            plt.title("Travelling Salesman Solution")
            plt.show()

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                result_string,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_hamiltonian_path_solution(input_data, sorted_vertices)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_travellingsalesman_solution.txt"
                if isinstance(input_data, str)
                else "travellingsalesman_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                result_string,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_hamiltonian_path_solution(input_data, sorted_vertices)
                ),
                txt_outputname=output_filename,
            )

        return sorted_vertices

    @staticmethod
    def independent_set(
        input_data: str | nx.Graph,
        a: float = 1,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the independent set problem for a given graph.

        Args:
            input_data (str | nx.Graph): Input data as file path or NetworkX graph.
            a (float): Penalty parameter for constraints. Default is 1.
            b (float): Penalty parameter for constraints. Default is 1.
            solve (bool): Whether to solve the problem or return a setup. Default is False.
            solver_method (Callable | None): Optional solver method.
            read_solution (Literal["print", "file"] | None): Output format for the solution.
            solver_params (dict | None): Additional parameters for the solver.
            visualize (bool): Whether to visualize the result graph. Default is False.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        a = b + 1

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {v: variables.add_binary_variable(f"x_{v}") for v in unique_vertices}

        ha = a * Add(*[x_vars[u] * x_vars[v] for u, v in edges])
        hb = -b * Add(*[x_vars[v] for v in unique_vertices])

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

        txt_outputname = "Maximal Independent Set: "

        solution = solver_method(problem, num_reads=500)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        independent_set = [int(var[len("x_") :]) for var, value in set_variables.items() if value == 1.0]

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            node_colors = ["lightgreen" if node in independent_set else "lightgrey" for node in graph.nodes()]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title("Maximal Independent Set")
            plt.show()

        output_set_1 = "Set 1: {" + ", ".join(map(str, independent_set)) + "}"

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_set_1,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_independent_set_solution(input_data, independent_set)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_independent_set_solution.txt"
                if isinstance(input_data, str)
                else "independent_set_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_set_1,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_independent_set_solution(input_data, independent_set)
                ),
                txt_outputname=output_filename,
            )

        return independent_set

    @staticmethod
    def check_independent_set_solution(graph: nx.Graph, solution: list[int]) -> dict[str, bool | dict[str, list]]:
        """Validates the solution for the independent set problem.

        Args:
            graph (nx.Graph): Input graph.
            solution (list[int]): Solution vertices in the independent set.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Invalid Nodes": [], "Conflicting Pairs": []}

        for node in solution:
            if node not in graph.nodes:
                errors["Invalid Nodes"].append(node)

        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if graph.has_edge(solution[i], solution[j]):
                    errors["Conflicting Pairs"].append((solution[i], solution[j]))

        if any(errors.values()):
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def _create_graph(clauses: list[list[str]]) -> nx.Graph:
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
            complement = literal[1:] if literal.startswith("¬") else "¬" + literal
            if complement in literals:
                g.add_edge(literal, complement)

        return g

    @staticmethod
    def _graph_to_text(g: nx.Graph) -> tuple[str, dict[str, str]]:
        """Converts a NetworkX graph to a textual representation suitable for file storage.

        Args:
            g (nx.Graph): The input graph to convert.

        Returns:
            tuple[str, dict[str, str]]: A tuple containing:
                - str: A string representation of the graph with vertices and edges.
                - dict[str, str]: A dictionary mapping vertex numbers to their original labels.
        """
        nodes = list(g.nodes)
        node_to_num = {node: i + 1 for i, node in enumerate(nodes)}

        edges = list(g.edges)
        edges_as_nums = [(node_to_num[u], node_to_num[v]) for u, v in edges]

        num_vertices = len(nodes)
        num_edges = len(edges_as_nums)

        result = f"{num_vertices} {num_edges}\n"
        for u, v in edges_as_nums:
            result += f"{u} {v}\n"

        num_to_node = {str(v): k for k, v in node_to_num.items()}

        return result, num_to_node

    @staticmethod
    def max_cut(
        input_data: str | nx.Graph,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: str | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the Max-Cut problem for a given graph, partitioning vertices to maximize the number of edges between partitions.

        Args:
            input_data (str | nx.Graph): Graph data as a file path or a NetworkX graph.
            solve (bool): If True, solves the problem; if False, returns the problem setup. Default is False.
            solver_method (Callable | None): Optional solver function.
            read_solution (str | None): Specifies output format ("print" or "file").
            solver_params (dict | None): Additional solver parameters.
            visualize (bool): If True, displays the solution graph. Default is False.

        Returns:
            Problem | list[int]: The formulated problem or, if solved, a list of vertices in one of the partitions.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            _num_vertices, num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for i in range(1, num_edges + 1):
                u, v = map(int, lines[i].strip().split())
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.Graph):
            filename = ""
            graph = input_data
            graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        problem = Problem()
        variables = Variables()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        x_vars = {v: variables.add_binary_variable(f"x_{v}") for v in unique_vertices}

        ha_terms = []

        sum_x_v = Add(*[2 * x_vars[u] * x_vars[v] - x_vars[u] - x_vars[v] for u, v in edges])
        ha_terms.append(sum_x_v)

        ha = simplify(Add(*ha_terms))

        objective_function.add_objective_function(ha)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Max-Cut: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("x_")}
        cut_set = [int(var[len("x_") :]) for var, value in set_variables.items() if value == 1.0]

        if visualize:
            graph = nx.Graph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            node_colors = ["lightgreen" if node in cut_set else "lightgray" for node in graph.nodes()]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True)

            plt.title("Max-Cut")
            plt.show()

        output_set = "S = {" + ", ".join(map(str, cut_set)) + "}"

        if read_solution == "print":
            KarpGraphs.print_solution(txt_outputname, filename, output_set)
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_max_cut_solution.txt"
                if isinstance(input_data, str)
                else "max_cut_solution.txt"
            )
            KarpGraphs.save_solution(txt_outputname, filename, output_set, txt_outputname=output_filename)

        return cut_set

    @staticmethod
    def directed_feedback_vertex_set(
        input_data: str | nx.DiGraph,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[int]:
        """Solves the directed feedback vertex set problem, identifying a minimal set of vertices.

        whose removal makes the directed graph acyclic.

        Args:
            input_data (str | nx.DiGraph): Directed graph data as a file path or NetworkX DiGraph.
            b (float): Penalty parameter for constraints. Default is 1.
            solve (bool): If True, solves the problem; if False, returns the problem setup. Default is False.
            solver_method (Callable | None): Optional solver function.
            read_solution (Literal["print", "file"] | None): Specifies output format.
            solver_params (dict | None): Additional solver parameters.
            visualize (bool): If True, displays the solution graph. Default is False.

        Returns:
            Problem | list[int]: The formulated problem or, if solved, a list of vertices in the feedback vertex set.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        unique_vertices = set()
        edges = []

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_vertices, _num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for line in lines[1:]:
                u, v = map(int, line.strip().split()[:2])
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.DiGraph):
            filename = ""
            graph = input_data
            num_vertices = graph.number_of_nodes()
            graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        a = b * 3.0

        unique_vertices = sorted(unique_vertices)
        {vertex: idx for idx, vertex in enumerate(unique_vertices, 1)}

        variables = Variables()
        y_vars = {v: variables.add_binary_variable(f"y_{v}") for v in unique_vertices}

        x_vars = {
            (v, i): variables.add_binary_variable(f"x_{v}_{i}")
            for v in unique_vertices
            for i in range(1, num_vertices + 1)
        }

        ha_terms = []

        for v in unique_vertices:
            term1 = a * (y_vars[v] - Add(*[x_vars[v, i] for i in range(1, num_vertices + 1)])) ** 2
            ha_terms.append(term1)

        for u, v in edges:
            for j in range(1, num_vertices + 1):
                term2 = a * Add(*[x_vars[u, i] * x_vars[v, j] for i in range(j, num_vertices + 1)])
                ha_terms.append(term2)

        ha = simplify(Add(*ha_terms))

        hb_terms = [b - b * y_vars[v] for v in unique_vertices]
        hb = simplify(Add(*hb_terms))

        h = ha + hb

        problem = Problem()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        objective_function.add_objective_function(h)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Feedback Vertex Set: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("y_")}
        feedback_vertex_set = [int(var[len("y_") :]) for var, value in set_variables.items() if value == 0.0]

        if visualize:
            graph = nx.DiGraph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            node_colors = ["red" if node in feedback_vertex_set else "lightgrey" for node in graph.nodes()]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, node_color=node_colors, with_labels=True, arrows=True)

            plt.title("Directed Feedback Vertex Set")
            plt.show()

        output_set = "F = {" + ", ".join(map(str, feedback_vertex_set)) + "}"

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_set,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_directed_feedback_vertex_set_solution(input_data, feedback_vertex_set)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_feedback_vertex_set_solution.txt"
                if isinstance(input_data, str)
                else "feedback_vertex_set_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_set,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_directed_feedback_vertex_set_solution(input_data, feedback_vertex_set)
                ),
                txt_outputname=output_filename,
            )

        return feedback_vertex_set

    @staticmethod
    def check_directed_feedback_vertex_set_solution(
        graph: nx.DiGraph, solution: list[int]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates the solution for the directed feedback edge vertex problem.

        Args:
            graph (nx.DiGraph): Input directed graph.
            solution (list[tuple[int, int]]): Solution edges in the feedback edge set.

        Returns:
            dict: Validation status with any errors found.
        """
        errors = {"Cycle Detected": []}

        modified_graph = graph.copy()
        modified_graph.remove_nodes_from(solution)

        is_acyclic = nx.is_directed_acyclic_graph(modified_graph)

        if not is_acyclic:
            try:
                cycle = list(nx.find_cycle(modified_graph, orientation="original"))
                errors["Cycle Detected"] = cycle
            except nx.exception.NetworkXNoCycle:
                pass

        if not is_acyclic:
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def directed_feedback_edge_set(
        input_data: str | nx.DiGraph,
        b: float = 1,
        solve: bool = False,
        solver_method: Callable | None = None,
        read_solution: Literal["print", "file"] | None = None,
        solver_params: dict | None = None,
        visualize: bool = False,
    ) -> Problem | list[tuple[int, int]]:
        """Validates the solution for the directed feedback edge set problem.

        Args:
            input_data (str | nx.DiGraph): Directed graph data as a file path or NetworkX DiGraph.
            b (float): Penalty parameter for constraints. Default is 1.
            solve (bool): If True, solves the problem; if False, returns the problem setup. Default is False.
            solver_method (Callable | None): Optional solver function.
            read_solution (Literal["print", "file"] | None): Specifies output format ("print" or "file").
            solver_params (dict | None): Additional parameters for the solver.
            visualize (bool): If True, displays the solution graph. Default is False.

        Returns:
            Problem | list[tuple[int, int]]: The formulated problem or, if solved, a list of edges in the feedback edge set.
        """
        if (
            any([solver_method is not None, read_solution is not None, solver_params is not None, visualize])
            and not solve
        ):
            msg = "'solve' must be True if 'solver_method', 'read_solution', 'solver_params', or 'visualize' are provided or True."
            raise ValueError(msg)

        if isinstance(input_data, str):
            filename = input_data
            try:
                with Path.open(filename) as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Error: File {input_data} not found.")
                return None

            num_vertices, _num_edges = map(int, lines[0].strip().split())
            edges = []
            unique_vertices = set()

            for line in lines[1:]:
                u, v = map(int, line.strip().split()[:2])
                edges.append((u, v))
                unique_vertices.update([u, v])

        elif isinstance(input_data, nx.DiGraph):
            filename = ""
            graph = input_data
            num_vertices = graph.number_of_nodes()
            graph.number_of_edges()
            edges = list(graph.edges())
            unique_vertices = set(graph.nodes())

        a = b * 3.0
        variables = Variables()

        y_vars = {(u, v): variables.add_binary_variable(f"y_{u}_{v}") for u, v in edges}

        x_vars = {
            (v, i): variables.add_binary_variable(f"x_{v}_{i}")
            for v in unique_vertices
            for i in range(1, num_vertices + 1)
        }

        xwu_vars = {
            (w, u, i): variables.add_binary_variable(f"x_{w}_{u}_{i}")
            for (w, u) in edges
            for i in range(1, num_vertices + 1)
        }

        ha_terms = []

        for v in unique_vertices:
            term1 = a * (1 - Add(*[x_vars[v, i] for i in range(1, num_vertices + 1)])) ** 2
            ha_terms.append(term1)

        for u, v in edges:
            term2 = a * (y_vars[u, v] - Add(*[xwu_vars[u, v, i] for i in range(1, num_vertices + 1)])) ** 2
            ha_terms.append(term2)

        for u, v in edges:
            for i in range(1, num_vertices + 1):
                term3 = a * (
                    xwu_vars[u, v, i]
                    * (2 - x_vars[u, i] - Add(*[x_vars[v, j] for j in range(i + 1, num_vertices + 1)]))
                )
                ha_terms.append(term3)

        ha = simplify(Add(*ha_terms))

        hb = b * Add(*[(1 - y_vars[u, v]) for u, v in edges])

        h = ha + hb

        problem = Problem()
        constraints = Constraints()
        objective_function = ObjectiveFunction()

        objective_function.add_objective_function(h)
        problem.create_problem(variables, constraints, objective_function)

        if not solve:
            return problem

        solver = Solver()
        if solver_method is None:
            solver_method = solver.solve_simulated_annealing
        if solver_params is not None:
            solver_method = partial(solver_method, **solver_params)

        txt_outputname = "Feedback Edge Set: "

        solution = solver_method(problem)
        set_variables = {k: v for k, v in solution.best_solution.items() if k.startswith("y_")}
        feedback_edge_set = [
            (int(var.split("_")[1]), int(var.split("_")[2])) for var, value in set_variables.items() if value == 0.0
        ]

        if visualize:
            graph = nx.DiGraph()
            graph.add_nodes_from(unique_vertices)
            graph.add_edges_from(edges)

            edge_colors = ["red" if (u, v) in feedback_edge_set else "black" for u, v in graph.edges()]

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, edge_color=edge_colors, with_labels=True, arrows=True)

            plt.title("Directed Feedback Edge Set")
            plt.show()

        output_set = "F = {" + ", ".join(f"({u},{v})" for u, v in feedback_edge_set) + "}"

        if read_solution == "print":
            KarpGraphs.print_solution(
                txt_outputname,
                filename,
                output_set,
                KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_directed_feedback_edge_set_solution(input_data, feedback_edge_set)
                ),
            )
        elif read_solution == "file":
            output_filename = (
                filename.replace(".txt", "") + "_feedback_edge_set_solution.txt"
                if isinstance(input_data, str)
                else "feedback_edge_set_solution.txt"
            )
            KarpGraphs.save_solution(
                txt_outputname,
                filename,
                output_set,
                summary=KarpGraphs.convert_dict_to_string(
                    KarpGraphs.check_directed_feedback_edge_set_solution(input_data, feedback_edge_set)
                ),
                txt_outputname=output_filename,
            )

        return feedback_edge_set

    @staticmethod
    def check_directed_feedback_edge_set_solution(
        graph: nx.DiGraph, solution: list[tuple[int, int]]
    ) -> dict[str, bool | dict[str, list]]:
        """Validates a solution for the directed feedback edge set problem, ensuring the removal.

        of specified edges results in an acyclic directed graph.

        Args:
            graph (nx.DiGraph): The input directed graph.
            solution (list[tuple[int, int]]): List of directed edges proposed for removal.

        Returns:
            dict: A dictionary with "Valid Solution" status (True/False) and details of any cycles
                  detected if the solution does not make the graph acyclic.
        """
        errors = {"Cycle Detected": []}

        modified_graph = graph.copy()
        modified_graph.remove_edges_from(solution)

        is_acyclic = nx.is_directed_acyclic_graph(modified_graph)

        if not is_acyclic:
            try:
                cycle = list(nx.find_cycle(modified_graph, orientation="original"))
                errors["Cycle Detected"] = cycle
            except nx.exception.NetworkXNoCycle:
                pass

        if not is_acyclic:
            return {"Valid Solution": False, "Errors": errors}

        return {"Valid Solution": True}

    @staticmethod
    def convert_dict_to_string(dictionary: dict) -> str:
        """Converts a validation dictionary to a formatted string.

        Args:
            dictionary (dict): Validation dictionary with solution status and errors.

        Returns:
            str: Formatted string representation of the dictionary.
        """
        result = "Valid Solution" if dictionary.get("Valid Solution", False) else "Invalid Solution"
        for key, value in dictionary.items():
            if key != "Valid Solution" and isinstance(value, dict):
                result += f"\n{key}:"
                for sub_key, sub_value in value.items():
                    if sub_value:
                        result += f"\n'{sub_key}': {sub_value}"
        return result
