from __future__ import annotations

import sys

sys.path.append("/mnt/data")
import os

# from src.mqt.qao.solvers import GroverOptimizerPolynomial
# from src import Constraints, ObjectiveFunction, Problem, Solver, Variables
import networkx as nx
import numpy as np

from src.mqt.karp.karp_graphs import Karp_Graphs
from src.mqt.karp.karp_number import Karp_Number

# from src.mqt.qao.problem import Problem
from src.mqt.karp.karp_sets import Karp_Sets
from src.mqt.qao.constraints import Constraints
from src.mqt.qao.problem import Problem
from src.mqt.qao.solvers import Solver
from src.mqt.qao.variables import Variables

# Suppress warnings globally
os.environ["PYTHONWARNINGS"] = "ignore::urllib3.exceptions.NotOpenSSLWarning"

import warnings

from urllib3.exceptions import NotOpenSSLWarning

# Suppress NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import sys

sys.path.append("/mnt/data")


def lambda_strategy(x):
    return x


def create_qubo():
    variables = Variables()
    constraint = Constraints()

    variables.add_binary_variable("a")
    variables.add_binary_variable("b")

    constraint.add_constraint("a & b")

    problem = Problem()
    problem.create_problem(variables, constraint)

    solver = Solver()
    solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

    if not isinstance(solution, bool):
        print("Best solution:", solution.best_solution)

    return problem


def test_qubo() -> None:
    qubo_problem = create_qubo()

    solver = Solver()
    solution = solver.solve_simulated_annealing(qubo_problem, lambda_strategy=lambda_strategy)

    if not isinstance(solution, bool):
        all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
        print("All constraints satisfied:", all_satisfy)
        print("Each constraint satisfaction:", each_satisfy)
        print("Best solution found:", solution.best_solution)
        # assert solution.best_solution == {"a": 0.0, "b": 4.0, "c": -1.5}
    else:
        print("No valid solution found")


def test_graph_coloring() -> None:
    input_content = """4 4
10 2
10 3
2 4
3 4
"""
    input_filename = "mock_graph_coloring_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        G2 = nx.Graph()
        nodes = range(1, 9)
        G2.add_nodes_from(nodes)
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1), (1, 3), (3, 5), (5, 7)]
        G2.add_edges_from(edges)

        problem = Karp_Graphs.graph_coloring(input_filename, num_colors=3)

        # print(problem)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

    finally:
        os.remove(input_filename)


def test_vertex_cover() -> None:
    input_content = """5 6
1 2
1 3
2 3
2 40
3 40
40 5
"""
    input_filename = "mock_vertex_cover_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.vertex_cover(input_filename, solve=True, output_file=False)
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        # assert len(problem.variables._variables_dict) == num_vertices

        # assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_clique() -> None:
    input_content = """5 7
100 2
100 3
2 3
2 400
3 400
400 5
100 400
"""
    input_filename = "mock_clique_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.clique(input_filename, K=3, solve=True, output_file=False)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        ##assert isinstance(problem, Problem)

        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_hamiltonian_path() -> None:
    input_content = """6 6
1 2
2 3
3 4
4 5
5 1
3 60
"""
    input_filename = "mock_hamiltonian_cycle_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem_path = Karp.hamiltonian_path(input_filename, cycle=False, solve=True, output_file=False)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem_path, lambda_strategy=lambda_strategy)
        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem_path, Problem)

        num_vertices = 5
        num_vertices * num_vertices
        ##assert len(problem_path.variables._variables_dict) == expected_num_variables

        ##assert len(problem_path.objective_function.objective_functions) == 1

        Karp.hamiltonian_path(input_filename, cycle=True)

        ##assert isinstance(problem_cycle, Problem)

        ##assert len(problem_cycle.variables._variables_dict) == expected_num_variables

        ##assert len(problem_cycle.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_independent_set() -> None:
    input_content = """4 4
10 2
10 3
2 400
3 400
"""
    input_filename = "mock_independent_set_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.independent_set(input_filename, solve=True, output_file=False)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_minimal_spanning_tree() -> None:
    input_content = """4 5
1 2 1
1 3 10
2 3 1
3 4 10
2 4 1
"""
    input_filename = "mock_minimal_spanning_tree_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.minimal_spanning_tree_new(input_filename, delta=2)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, num_reads=100000)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_vertices = 5
        num_edges = 6
        Delta = 3
        2 * num_edges + num_vertices * num_vertices * (num_vertices // 2 + 1) + num_vertices * (Delta + 1)
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_integer_programming() -> None:
    input_content = """3 2
1 0 1
0 1 1
1 20
3 4 5
"""
    input_filename = "mock_integer_programming_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp_Number.integer_programming(input_filename, solve=True, read_solution="print")
        print(problem)
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_knapsack() -> None:
    input_content = """5 10
4 40
6 30
3 50
2 10
"""
    input_filename = "mock_knapsack_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        input_data = [(2, 3), (1, 2), (3, 4), (2, 4)]
        problem = Karp_Number.knapsack(input_data, num_objects=5, max_weight=10, solve=True, output_type="print")
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_objects = 5
        max_weight = 10
        num_objects + max_weight
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_set_problems() -> None:
    input_content = """5 3
20 2 1 2
200 2 2 3
60 2 4 5
"""
    input_filename = "20mock_inputt.txt"

    input_file_content = """6 6
10 1
1 2
3 4
4 2
4 5
1 5
"""
    inputfile = "mock_inpuhgdghdft.txt"

    with open(inputfile, "w", encoding="utf-8") as file:
        file.write(input_file_content)

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    input_file_content = """6 6
10 1
1 2
3 4
4 2
4 5
1 5
"""
    inputfile_3d = "mock_input_3d_matching.txt"
    with open(inputfile_3d, "w", encoding="utf-8") as file:
        file.write(input_file_content)

    try:
        sets_list = [(20, [1, 2]), (3, [3232323232, 3]), (4, [55, 5])]
        # Example call
        # 1
        print(Karp_Sets.hitting_set(sets_list, solve=True, read_solution="print"))
        # 2
        # Karp_Sets.set_cover('mock_input.txt', solve=True, output_file=False, weighted=True)
        # 3
        # Karp_Sets.set_packing(input_filename, solve=True, output_type="print")
        # 4
        # print(Karp_Sets.set_cover(sets_list, solve=True, output_type="print"))
        # 5

        G = nx.Graph()

        # Adding edges to the graph (example graph)
        G = nx.Graph()
        G.add_edges_from([
            (5, 8),
            (1, 8),
            (2, 9),
            (2, 10),
            (3, 11),
            (3, 12),
            (4, 7),
            (4, 11),
            (5, 11),
            (5, 16),
            (6, 17),
            (6, 18),
            (7, 19),
            (8, 20),
            (9, 21),
            (10, 22),
            (11, 23),
            (12, 24),
            (13, 25),
            (9, 19),
            (8, 24),
            (16, 28),
            (17, 29),
            (18, 30),
            (19, 31),
            (20, 32),
            (21, 33),
            (22, 34),
            (23, 35),
            (24, 36),
            (25, 37),
            (26, 38),
            (27, 39),
            (28, 40),
            (29, 41),
            (30, 42),
        ])

        # Define the sets X, Y, Z

        # Run the three_d_matching function
        # print(Karp_Sets.three_d_matching(G, X, Y, Z, solve=True, read_solution="print", visualise=False))
        # Karp_Sets.hitting_set('20mock_inputt.txt', solve=True, output_file=False)

    finally:
        os.remove(input_filename)

    """
    Perform the reduction from Feedback Arc Set to Feedback Node Set.

    Parameters:
    G (nx.DiGraph): A directed graph where the reduction will be performed.

    Returns:
    nx.Graph: The line graph representing the reduction, where the Feedback Node Set
              problem can be applied.
    """
    # Step 1: Create an expanded graph G'


def feedback_arc_set_to_feedback_node_set(G):
    """
    Perform the reduction from Feedback Arc Set to Feedback Node Set.

    Parameters:
    G (nx.DiGraph): A directed graph where the reduction will be performed.

    Returns:
    nx.DiGraph: The line graph representing the reduction, where the Feedback Node Set
                problem can be applied with integer node labels.
    dict: Mapping from integer nodes in the line graph to the original expanded graph edges (tuples).
    """
    G_prime = nx.DiGraph()
    expanded_nodes = {}
    node_counter = 0

    # Create expanded nodes in G' and map them to integers
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        path_length = in_degree + out_degree

        # Create a path of 'path_length' nodes in G_prime
        path_nodes = list(range(node_counter, node_counter + path_length))
        for i in range(len(path_nodes) - 1):
            G_prime.add_edge(path_nodes[i], path_nodes[i + 1])

        # Map the original node to its expanded nodes in G_prime
        expanded_nodes[node] = path_nodes
        node_counter += path_length

    # Add edges corresponding to the original graph G
    edge_to_expanded_edge_map = {}
    for u, v in G.edges():
        u_expanded = expanded_nodes[u][-1]  # Last node in the expanded path for u
        v_expanded = expanded_nodes[v][0]  # First node in the expanded path for v
        G_prime.add_edge(u_expanded, v_expanded)

        # Map original edge to the corresponding edge in G'
        edge_to_expanded_edge_map[u, v] = (u_expanded, v_expanded)

    # Create the line graph of G_prime
    line_G_prime = nx.line_graph(G_prime)

    # Create a mapping from line graph nodes (integers) to expanded graph edges (tuples)
    line_node_to_edge_map = {}
    line_graph_with_ints = nx.DiGraph()
    for i, edge in enumerate(line_G_prime.nodes()):
        line_node_to_edge_map[i] = edge
        line_graph_with_ints.add_node(i)

    # Add edges to the new line graph with integer nodes
    for edge in line_G_prime.edges():
        node_u = next(k for k, v in line_node_to_edge_map.items() if v == edge[0])
        node_v = next(k for k, v in line_node_to_edge_map.items() if v == edge[1])
        line_graph_with_ints.add_edge(node_u, node_v)

    return line_graph_with_ints, line_node_to_edge_map, expanded_node_to_original_node_map


def interpret_feedback_vertex_set(feedback_edge_set, line_node_to_edge_map, expanded_node_to_original_node_map):
    original_vertices = set()
    for edge in feedback_edge_set:
        u_expanded, v_expanded = line_node_to_edge_map[edge]
        original_vertices.add(expanded_node_to_original_node_map[u_expanded])
        original_vertices.add(expanded_node_to_original_node_map[v_expanded])
    return original_vertices


def test_graphs() -> None:
    input_content = """6 7
1 2
1 3
2 3
3 4
4 5
5 6
6 4
"""
    input_filename = "mock_clique_cover_inkut.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        # problem = Karp.clique_cover(input_filename, num_colors=2, solve=True, output_file=False)
        G_1 = nx.Graph()

        # Add edges to the graph
        G_1.add_edge(1, 2)
        G_1.add_edge(1, 3)
        G_1.add_edge(1, 5)
        G_1.add_edge(2, 6)
        G_1.add_edge(3, 7)
        G_1.add_edge(4, 8)
        G_1.add_edge(5, 6)
        G_1.add_edge(7, 6)
        G_1.add_edge(7, 8)
        G_1.add_edge(5, 8)
        G_1.add_edge(3, 4)
        G_1.add_edge(2, 4)

        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(range(1, 7))  # Adding nodes 1 to 6

        # Define cliques with weighted edges
        clique1 = [(1, 2, 2.5), (2, 3, 3.7), (3, 4, 1.2), (4, 5, 4.0), (5, 1, 2.1), (1, 3, 3.0), (6, 1, 2.8)]
        clique2 = [(5, 6, 1.5)]

        # Add edges for each clique
        G.add_weighted_edges_from(clique1)
        G.add_weighted_edges_from(clique2)

        G2 = nx.Graph()

        nodes = range(1, 9)
        G2.add_nodes_from(nodes)

        # Add edges
        edges = [(1, 2), (2, 3), (3, 4)]
        G2.add_edges_from(edges)
        # 6
        # print(Karp_Graphs.clique_cover(G, solve=True, num_colors=3, read_solution="print", visualize=True))
        # 7
        # print(Karp_Graphs.hamiltonian_path(G, solve=True, cycle=True, read_solution="print", visualize=True))
        # 8
        # print(Karp_Graphs.clique(G, K=0, solve=True, read_solution="print", visualize=True))
        # 9
        # print(Karp_Graphs.vertex_cover(G, solve=True, read_solution="print", visualize=True))
        # 10
        # print(Karp_Graphs.clique_cover(G, num_colors=4, solve=True, read_solution="print", visualize=True))
        # 11
        # print(Karp_Graphs.max_cut(G, solve=True, read_solution="print", visualize=False))
        # 12
        # Karp_Graphs.graph_coloring(G, num_colors=3, solve=True, read_solution="print", visualize=True)

        # Karp_Graphs.independent_set(G, solve=True, read_solution="print", visualize=True)

        G2_directed = nx.DiGraph()
        G2_directed.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (4, 5)])
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(range(1, 7))  # Adding nodes 1 to 6

        # Define cliques with additional edges to form cycles
        clique1 = [(1, 2), (2, 3), (4, 3), (5, 4), (1, 6), (6, 5)]  # Original edges
        clique2 = [(5, 1), (3, 1)]  # Additional edges forming directed cycles

        # Add directed cycles (new edges that introduce cycles)

        # Add directed edges for each clique and the cycles
        G.add_edges_from(clique1)
        G.add_edges_from(clique2)

        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))

        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))
        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))
        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))
        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))

        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))
        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))
        # print(Karp_Graphs.directed_feedback_edge_set(G, solve=True, read_solution="print", visualize= True))

        # 13
        # print(Karp_Graphs.directed_feedback_edge_set(graph, solve=True, read_solution="print", visualize=True))
        # print(interpret_feedback_vertex_set(Karp_Graphs.directed_feedback_edge_set(graph, solve=True, read_solution="print", visualize=True), ))
        input_content = """6 8
1 2 1
2 3 20
3 4 1
4 5 20
5 1 1
1 6 1
5 6 1
1 3 1
"""
        input_filename = "mock_travelling_salesman_input4t.txt"

        with open(input_filename, "w", encoding="utf-8") as file:
            file.write(input_content)
        # 14
        # print(Karp_Graphs.travelling_salesman(G, solve=True, cycle=False, read_solution="print", visualize=True))

        problem = Karp_Graphs.graph_coloring(G, num_colors=3)

        solver = Solver()

        # Solve the problem using simulated annealing

        # assert isinstance(problem, Problem)

        num_vertices = 4
        num_colors = 1
        num_vertices * num_colors
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

        import matplotlib.pyplot as plt

        # Remove LaTeX settings if LaTeX is not available
        # rc('text', usetex=True)
        # plt.rc('text', usetex=True)
        # plt.rcParams.update({'font.size': 20, 'text.usetex': True})

        # Rest of your plotting code
        # Assumiimport matplotlib.pyplot as plt

        # Define the problem
        problem = Karp_Graphs.graph_coloring(G, num_colors=3)

        # Instantiate the solver
        solver = Solver()

        # List of parameter variations to test
        parameter_variations = [
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "label": "Default SA",
            },
            {
                "num_reads": 200,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "label": "Increased Reads",
            },
            {
                "num_reads": 100,
                "annealing_time": 500,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "label": "Increased Annealing Time",
            },
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.5, 2.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "label": "Tighter Beta Range",
            },
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "linear",
                "label": "Linear Beta Schedule",
            },
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 5,
                "beta_schedule_type": "geometric",
                "label": "Increased Sweeps Per Beta",
            },
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "initial_states_generator": "random",
                "label": "Random Initial States",
            },
            {
                "num_reads": 100,
                "annealing_time": 100,
                "beta_range": [0.1, 4.0],
                "num_sweeps_per_beta": 1,
                "beta_schedule_type": "geometric",
                "lambda_update_mechanism": "sequential penalty increase",
                "label": "Sequential Penalty Increase",
            },
        ]

        # Reference optimal energy (you can adjust this based on your problem)

        # Initialize figure for plotting all variations
        plt.figure(figsize=(10, 6))

        # Iterate over each parameter variation
        for params in parameter_variations:
            print(f"1. Running Simulated Annealing with parameters: {params['label']}")

            # Solve the problem using Simulated Annealing with the current parameter variation
            solution_sa = solver.solve_simulated_annealing(
                problem,
                num_reads=params.get("num_reads", 100),
                annealing_time=params.get("annealing_time", 100),
                beta_range=params.get("beta_range", [0.1, 4.0]),
                num_sweeps_per_beta=params.get("num_sweeps_per_beta", 1),
                beta_schedule_type=params.get("beta_schedule_type", "geometric"),
                initial_states_generator=params.get("initial_states_generator", "random"),
                lambda_update_mechanism=params.get("lambda_update_mechanism", None),
            )

            # Check if solution was found
            if solution_sa:
                # Extract the energies from the Simulated Annealing solution
                EnSimulated = solution_sa.energies

                # Plot the cumulative distribution of Simulated Annealing energies
                _n, _bins, _patches = plt.hist(
                    EnSimulated, cumulative=True, histtype="step", linewidth=2, bins=100, label=params["label"]
                )

            else:
                print(f"Simulated Annealing with parameters '{params['label']}' did not find a solution.")

        # Add a vertical line showing the optimal energy reference point
        # plt.axvline(x=ref, color='tab:red', linewidth=4, label='Optimal Energy')

        # Adding labels and title
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Occurrence", fontsize=20)
        plt.title("Simulated Annealing Cumulative Distribution for Various Parameters", fontsize=22)

        # Add legend
        plt.legend(loc="lower right", frameon=True, fontsize=12)

        # Save the plot to different formats
        plt.savefig("CumulativeSAExample2.eps", format="eps", bbox_inches="tight")
        plt.savefig("CumulativeSAExample2.png", format="png", bbox_inches="tight")
        plt.savefig("CumulativeSAExample2.pdf", format="pdf", bbox_inches="tight")

        # Show the plot
        plt.show()

    finally:
        os.remove(input_filename)


def create_fully_connected_graph(n):
    """
    Creates a fully connected graph (complete graph) with n vertices.

    Parameters:
    n (int): Number of vertices in the graph

    Returns:
    G (networkx.Graph): A fully connected graph with n vertices
    """
    # Create a complete graph with n vertices
    return nx.complete_graph(n)

    # Optionally, you can plot the graph to visualize it


def test_number_partition() -> None:
    input_content = """10
5
3
1
1
3
1
2
2
1
1
"""
    input_filename = "mock_number_partition_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        # 14
        print(Karp_Number.number_partition([3, 1, 2, 2, 4], solve=True, read_solution="print", visualize=True))
        # 15
        # input_data = [(2, 10), (55, 2), (3, 4), (2, 4)]
        # result = Karp_Number.knapsack(input_data, max_weight=1, solve=True, output_type="print")
        # 16

        Solver()
        # solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_travelling_salesman() -> None:
    input_content = """4 6
1 2 1
1 3 20
2 3 1
2 4 20
3 4 1
1 4 1
"""
    input_filename = "mock_travelling_salesman_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        graph = nx.Graph()

        # Add nodes
        graph.add_nodes_from([1, 2, 3, 4, 5, 6])

        # Add edges with weights
        edges = [
            (1, 2, {"weight": 7}),
            (1, 3, {"weight": 9}),
            (1, 4, {"weight": 14}),
            (1, 5, {"weight": 10}),
            (1, 6, {"weight": 15}),
            (2, 3, {"weight": 10}),
            (2, 4, {"weight": 14}),
            (2, 5, {"weight": 15}),
            (2, 6, {"weight": 9}),
            (3, 4, {"weight": 11}),
            (3, 5, {"weight": 12}),
            (3, 6, {"weight": 6}),
            (4, 5, {"weight": 2}),
            (4, 6, {"weight": 9}),
            (5, 6, {"weight": 6}),
        ]
        graph.add_edges_from(edges)
        problem = Karp_Graphs.travelling_salesman(graph)
        print(problem)
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        solver = Solver()

        # Solve the problem using simulated annealing
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        # Check if a solution was found, then call the show_cumulative method
        if solution:
            solution.show_cumulative(
                save=False,  # Change to True if you want to save the graph
                show=True,  # Change to False if you don't want to display the graph
                filename="solution_cumulative",  # Filename to save (only if save=True)
                label="TSP Solution",  # Optional label for the plot
                latex=False,  # Change to True if you want to use LaTeX font in the plot
            )

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_vertices = 4
        num_vertices * num_vertices
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_job_sequencing() -> None:
    input_content = """4
1
2
2
2
"""
    input_filename = "mock_job_sequencing_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp_Number.job_sequencing(input_filename, m=3, solve=True, read_solution="print")
        print(problem)
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_jobs = 4
        m = 3
        num_jobs * m + (m - 1) * (num_jobs + 1)
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_max_cut() -> None:
    input_content = """5 6
10 20
10 30
20 30
20 40
30 50
40 50
"""
    input_filename = "mock_clique_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.max_cut(input_filename, solve=True, output_file=False)

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_hitting_set() -> None:
    input_content = """3 3
20 2 1000
20 1 40
60 1 3
"""
    input_filename = "mock_hitting_set_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        sets_list = [(20, [3, 10]), (3, [400000]), (4, [1])]
        problem = Karp_Sets.hitting_set(sets_list, solve=True, output_type="print")

        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_elements = 3
        num_sets = 4
        num_sets + (num_elements * num_sets)
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_directed_feedback_vertex_and_edge_set() -> None:
    input_content = """4 5
10 20 1
20 10 1
20 3 1
3 4 1
4 10 1
"""
    input_filename = "mock_directed_feedback_set_input.txt"

    with open(input_filename, "w", encoding="utf-8") as file:
        file.write(input_content)

    try:
        problem = Karp.directed_feedback_vertex_set(input_filename, solve=True, output_file=False)
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

        # assert isinstance(problem, Problem)

        num_vertices = 4
        expected_num_y_vars = num_vertices
        expected_num_x_vars = num_vertices * num_vertices
        expected_num_y_vars + expected_num_x_vars
        ##assert len(problem.variables._variables_dict) == expected_num_variables

        ##assert len(problem.objective_function.objective_functions) == 1

    finally:
        os.remove(input_filename)


def test_3d_matching() -> None:
    input_file_content = """6 6
1 3
3 5
2 4
4 6
1 4
4 6
"""
    X = [1, 2]
    Y = [3, 4]
    Z = [5, 6]
    inputfile = "mock_input_3d_matching.txt"
    with open(inputfile, "w", encoding="utf-8") as file:
        file.write(input_file_content)

    try:
        problem = Karp_Sets.three_d_matching(inputfile, X, Y, Z, solve=True, read_solution="print", visualize=True)
        print("set cover")
        solver = Solver()
        solution = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)

        if not isinstance(solution, bool):
            print("Best solution:", solution.best_solution)

    finally:
        os.remove(inputfile)


def create_graph(clauses):
    G = nx.Graph()

    # Create a node for each literal in each clause
    for clause in clauses:
        for literal in clause:
            G.add_node(literal)

    # Connect the three literals in each clause to form a triangle
    for clause in clauses:
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                G.add_edge(clause[i], clause[j])

    # Connect complementary literals
    literals = {literal for clause in clauses for literal in clause}
    for literal in literals:
        complement = literal[1:] if literal.startswith("¬") else "¬" + literal
        if complement in literals:
            G.add_edge(literal, complement)

    return G


def create_graph(clauses):
    G = nx.Graph()

    # Create a node for each literal in each clause
    for clause in clauses:
        for literal in clause:
            G.add_node(literal)

    # Connect the three literals in each clause to form a triangle
    for clause in clauses:
        for i in range(len(clause)):
            for j in range(i + 1, len(clause)):
                G.add_edge(clause[i], clause[j])

    # Connect complementary literals
    literals = {literal for clause in clauses for literal in clause}
    for literal in literals:
        complement = literal[1:] if literal.startswith("¬") else "¬" + literal
        if complement in literals:
            G.add_edge(literal, complement)

    return G


def graph_to_text(G):
    nodes = list(G.nodes)
    node_to_num = {node: i + 1 for i, node in enumerate(nodes)}

    edges = list(G.edges)
    edges_as_nums = [(node_to_num[u], node_to_num[v]) for u, v in edges]

    num_vertices = len(nodes)
    num_edges = len(edges_as_nums)

    result = f"{num_vertices} {num_edges}\n"
    for u, v in edges_as_nums:
        result += f"{u} {v}\n"

    return result, node_to_num


def test_set_problems() -> None:
    print("1. Testing Set Cover Problem")
    set_cover_input = [(1, [1, 2, 3]), (1, [3, 4]), (1, [3, 4, 5]), (1, [3, 5])]
    set_cover_result = Karp_Sets.set_cover(set_cover_input, solve=True, read_solution="print", weighted=False)
    print("Set Cover Result:", set_cover_result)
    print("\n")

    print("2. Testing Set Packing Problem")
    # set_packing_input = [(3, [1, 2]), (2, [3, 4]), (1, [1, 4])]
    # set_packing_result = Karp_Sets.set_packing(set_packing_input, solve=True, read_solution="print")
    # print("Set Packing Result:", set_packing_result)
    print("\n")

    print("3. Testing Hitting Set Problem")
    # hitting_set_input = [(2, [1, 3]), (1, [2, 4]), (3, [1, 4])]
    # hitting_set_result = Karp_Sets.hitting_set(hitting_set_input, solve=True, read_solution="print")
    # print("Hitting Set Result:", hitting_set_result)
    print("\n")

    print("4. Testing 3D Matching Problem")
    graph = nx.Graph()
    graph.add_edges_from([(1, 3), (3, 5), (2, 4), (4, 6), (1, 4), (4, 6)])
    X = [1, 2]
    Y = [3, 4]
    Z = [5, 6]
    Karp_Sets.three_d_matching(graph, X, Y, Z, solve=True, visualize=True, read_solution="print")
    # print("3D Matching Result:", three_d_matching_result)
    print("\n")

    print("5. Testing Exact Cover Problem")
    # exact_cover_input = [(2, [1, 4]), (1, [2, 3]), (3, [1, 3, 4])]
    # exact_cover_result = Karp_Sets.exact_cover(exact_cover_input, solve=True, read_solution="print")
    # print("Exact Cover Result:", exact_cover_result)


import matplotlib.pyplot as plt


# Function to create a fully connected graph with n vertices
def create_fully_connected_graph(n):
    """
    Creates a fully connected graph (complete graph) with n vertices.

    Parameters:
    n (int): Number of vertices in the graph

    Returns:
    G (networkx.Graph): A fully connected graph with n vertices
    """
    return nx.complete_graph(n)


# Simulated Annealing experiment
def run_simulated_annealing_experiment(num_graphs) -> None:
    """
    Runs Simulated Annealing on num_graphs fully connected graphs and tests different parameter variations.

    Parameters:
    num_graphs (int): Number of fully connected graphs to run the experiment on
    """
    solver = Solver()

    # List of parameter variations to test
    parameter_variations = [
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Default SA",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 2,
        },
        # {"num_reads": 200, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Increased Reads", "color": 'tab:orange', "linestyle": '-', "linewidth": 1.5},
        {
            "num_reads": 100,
            "annealing_time": 200,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Increased Annealing Time",
            "color": "tab:green",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.01, 20.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Wider Beta Range",
            "color": "tab:red",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "linear",
            "label": "Linear Beta Schedule",
            "color": "tab:purple",
            "linestyle": ":",
            "linewidth": 1,
            "alpha": 0.5,
        },
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 5,
            "beta_schedule_type": "geometric",
            "label": "Increased Sweeps Per Beta",
            "color": "tab:brown",
            "linestyle": "-",
            "linewidth": 1,
            "alpha": 0.5,
        },
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "initial_states_generator": "random",
            "label": "Random Initial States",
            "color": "tab:pink",
            "linestyle": "-",
            "linewidth": 1,
            "alpha": 0.5,
        },
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "lambda_update_mechanism": "sequential penalty increase",
            "label": "Sequential Penalty Increase",
            "color": "tab:gray",
            "linestyle": ":",
            "linewidth": 1,
            "alpha": 0.9,
        },
        # {"num_reads": 200, "annealing_time": 200, "beta_range": [0.01, 20.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Combined", "color": 'purple', "linestyle": '-', "linewidth": 2, "alpha": 0.9}
    ]

    # Reference optimal energy (you can adjust this based on your problem)

    # Iterate over the number of graphs
    for i in range(3, 18, 3):
        print(f"\nRunning experiment for fully connected graph with {i} vertices")

        # Create a fully connected graph with i vertices
        G = create_fully_connected_graph(i)

        # Define the problem using the generated graph (modify this for your specific problem)

        # Iterate over each parameter variation
        for params in parameter_variations:
            print(f"1. Running Simulated Annealing with parameters: {params['label']} on graph with {i} vertices")
            problem = Karp_Graphs.graph_coloring(G, num_colors=i)
            solver = Solver()

            # Solve the problem using Simulated Annealing with the current parameter variation
            solution_sa = solver.solve_simulated_annealing(
                problem,
                num_reads=params.get("num_reads", 100),
                annealing_time=params.get("annealing_time", 100),
                beta_range=params.get("beta_range", [0.1, 4.0]),
                num_sweeps_per_beta=params.get("num_sweeps_per_beta", 1),
                beta_schedule_type=params.get("beta_schedule_type", "geometric"),
                initial_states_generator=params.get("initial_states_generator", "random"),
                lambda_update_mechanism=params.get("lambda_update_mechanism", None),
            )

            if solution_sa:
                EnSimulated = solution_sa.energies

                # Optional: Add a slight jitter to the energies to prevent exact overlaps
                # EnSimulated = [e + np.random.uniform(-0.01, 0.01) for e in EnSimulated]

                # Plot the cumulative distribution of Simulated Annealing energies
                _n, _bins, _patches = plt.hist(
                    EnSimulated,
                    cumulative=True,
                    histtype="step",
                    color=params["color"],
                    linewidth=params["linewidth"],
                    bins=100,
                    label=params["label"],
                    linestyle=params["linestyle"],
                    alpha=params.get("alpha", 1),
                )
            else:
                print(f"Simulated Annealing with parameters '{params['label']}' did not find a solution.")

        # plt.axvline(x=ref, color='tab:red', linewidth=4, label='Optimal Energy')

        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Occurrence", fontsize=20)
        plt.title(f"{i} Vertices", fontsize=20)

        # plt.legend(loc='lower right', frameon=True, fontsize=10)

        plt.savefig(f"graphcoloring_{i}_Vertices.png", format="png", bbox_inches="tight")
        plt.close()


# Example usage: Run the experiment for graphs with 1 to 5 vertices


def generate_random_problem(n):
    # Generate a random n x n matrix S with values in a specified range, e.g., 0 to 10
    S = np.random.randint(0, 2, size=(n, n))

    x = np.random.randint(0, 2, size=n)

    # Calculate vector b = S @ x (matrix-vector multiplication)
    b = S @ x

    # Set the vector c to a zero vector (since c is ignored)
    c = np.zeros(n, dtype=int)

    # Combine S, b, and c into a single input list for the solver
    return np.vstack([S, b, c]).tolist()


# Simulated Annealing experiment
def run_simulated_annealing_experiment_integer(num_graphs) -> None:
    """
    Runs Simulated Annealing on num_graphs fully connected graphs and tests different parameter variations.

    Parameters:
    num_graphs (int): Number of fully connected graphs to run the experiment on
    """

    # List of parameter variations to test
    parameter_variations = [
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Default SA",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        # {"num_reads": 400, "annealing_time": 500, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label":"Increased Reads", "color": 'tab:orange', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 200, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Increased Annealing Time", "color": 'tab:green', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.01, 20.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Wider Beta Range", "color": 'tab:red', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "linear", "label": "Linear Beta Schedule", "color": 'tab:purple', "linestyle": ':', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 5, "beta_schedule_type": "geometric", "label": "Increased Sweeps Per Beta", "color": 'tab:brown', "linestyle": '-', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "initial_states_generator": "random", "label": "Random Initial States", "color": 'tab:pink', "linestyle": '-', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "lambda_update_mechanism": "sequential penalty increase", "label": "Sequential Penalty Increase", "color": 'tab:gray', "linestyle": ':', "linewidth": 1, "alpha": 0.9},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Default SA1", "color": 'tab:blue', "linestyle": '-', "linewidth": 1.5}
        {
            "num_reads": 200,
            "annealing_time": 200,
            "beta_range": [0.01, 20.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Combined",
            "color": "purple",
            "linestyle": "-",
            "linewidth": 2,
            "alpha": 0.9,
        },
    ]

    # Reference optimal energy (you can adjust this based on your problem)

    # Iterate over the number of graphs
    for i in range(12, 30, 2):
        print(f"\nRunning experiment for integer {i}xi")

        i = 30

        G = generate_random_problem(i)

        # Define the problem using the generated graph (modify this for your specific problem)

        # Iterate over each parameter variation
        for params in parameter_variations:
            print(params)
            print(f"1. Running Simulated Annealing with parameters: {params['label']} on graph with {i} vertices")
            solver = Solver()
            problem = Karp_Number.integer_programming(input_data=G)

            # Solve the problem using Simulated Annealing with the current parameter variation
            solution_sa = solver.solve_simulated_annealing(
                problem,
                num_reads=params.get("num_reads", 100),
                annealing_time=params.get("annealing_time", 100),
                beta_range=params.get("beta_range", [0.1, 4.0]),
                num_sweeps_per_beta=params.get("num_sweeps_per_beta", 1),
                beta_schedule_type=params.get("beta_schedule_type", "geometric"),
                initial_states_generator=params.get("initial_states_generator", "random"),
                lambda_update_mechanism=params.get("lambda_update_mechanism", None),
            )

            if solution_sa:
                EnSimulated = solution_sa.energies

                # Optional: Add a slight jitter to the energies to prevent exact overlaps
                # EnSimulated = [e + np.random.uniform(-0.01, 0.01) for e in EnSimulated]

                # Plot the cumulative distribution of Simulated Annealing energies
                _n, _bins, _patches = plt.hist(
                    EnSimulated,
                    cumulative=True,
                    histtype="step",
                    color=params["color"],
                    linewidth=params["linewidth"],
                    bins=100,
                    label=params["label"],
                    linestyle=params["linestyle"],
                    alpha=params.get("alpha", 1),
                )
            else:
                print(f"Simulated Annealing with parameters '{params['label']}' did not find a solution.")

        # plt.axvline(x=ref, color='tab:red', linewidth=4, label='Optimal Energy')

        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Occurrence", fontsize=20)
        plt.title(f"{i}x{i}", fontsize=20)

        # plt.legend(loc='lower right', frameon=True, fontsize=7)

        plt.savefig(f"MATRIX2_{i}_SA.png", format="png", bbox_inches="tight")
        plt.close()
        break


def create_sets(n):
    sets = []
    i = 1
    while i + 1 <= n / 2:
        sets.append([i, i + 1])
        i += 2
    if i <= n:
        sets.append(list(range(i, n + 1)))

    while i <= n:
        sets.append([i])
        i += 1
    # random.shuffle(sets)

    return [(1, s) for s in sets]


# Example usage:
n = 12
result = create_sets(n)
print(result)


def run_simulated_annealing_experiment_sets(num_graphs) -> None:
    """
    Runs Simulated Annealing on num_graphs fully connected graphs and tests different parameter variations.

    Parameters:
    num_graphs (int): Number of fully connected graphs to run the experiment on
    """

    # List of parameter variations to test
    parameter_variations = [
        {
            "num_reads": 100,
            "annealing_time": 100,
            "beta_range": [0.1, 4.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Default SA",
            "color": "tab:blue",
            "linestyle": "-",
            "linewidth": 1.5,
        },
        # {"num_reads": 400, "annealing_time": 500, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label":"Increased Reads", "color": 'tab:orange', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 200, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Increased Annealing Time", "color": 'tab:green', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.01, 20.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Wider Beta Range", "color": 'tab:red', "linestyle": '-', "linewidth": 1.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "linear", "label": "Linear Beta Schedule", "color": 'tab:purple', "linestyle": ':', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 5, "beta_schedule_type": "geometric", "label": "Increased Sweeps Per Beta", "color": 'tab:brown', "linestyle": '-', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "initial_states_generator": "random", "label": "Random Initial States", "color": 'tab:pink', "linestyle": '-', "linewidth": 1, "alpha": 0.5},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "lambda_update_mechanism": "sequential penalty increase", "label": "Sequential Penalty Increase", "color": 'tab:gray', "linestyle": ':', "linewidth": 1, "alpha": 0.9},
        # {"num_reads": 100, "annealing_time": 100, "beta_range": [0.1, 4.0], "num_sweeps_per_beta": 1, "beta_schedule_type": "geometric", "label": "Default SA1", "color": 'tab:blue', "linestyle": '-', "linewidth": 1.5}
        {
            "num_reads": 200,
            "annealing_time": 200,
            "beta_range": [0.01, 20.0],
            "num_sweeps_per_beta": 1,
            "beta_schedule_type": "geometric",
            "label": "Combined",
            "color": "purple",
            "linestyle": "-",
            "linewidth": 2,
            "alpha": 0.9,
        },
    ]

    # Reference optimal energy (you can adjust this based on your problem)

    # Iterate over the number of graphs
    for i in range(12, 30, 2):
        print(f"\nRunning experiment for integer {i}xi")

        i = 5

        G = create_sets(i)

        # Define the problem using the generated graph (modify this for your specific problem)

        # Iterate over each parameter variation
        for params in parameter_variations:
            print(params)
            print(f"1. Running Simulated Annealing with parameters: {params['label']} on graph with {i} vertices")
            solver = Solver()
            problem = Karp_Sets.exact_cover(input_data=G, B=1)

            # Solve the problem using Simulated Annealing with the current parameter variation
            solution_sa = solver.solve_simulated_annealing(
                problem,
                num_reads=params.get("num_reads", 100),
                annealing_time=params.get("annealing_time", 100),
                beta_range=params.get("beta_range", [0.1, 4.0]),
                num_sweeps_per_beta=params.get("num_sweeps_per_beta", 1),
                beta_schedule_type=params.get("beta_schedule_type", "geometric"),
                initial_states_generator=params.get("initial_states_generator", "random"),
                lambda_update_mechanism=params.get("lambda_update_mechanism", None),
            )

            if solution_sa:
                EnSimulated = solution_sa.energies

                # Optional: Add a slight jitter to the energies to prevent exact overlaps
                # EnSimulated = [e + np.random.uniform(-0.01, 0.01) for e in EnSimulated]

                # Plot the cumulative distribution of Simulated Annealing energies
                _n, _bins, _patches = plt.hist(
                    EnSimulated,
                    cumulative=True,
                    histtype="step",
                    color=params["color"],
                    linewidth=params["linewidth"],
                    bins=100,
                    label=params["label"],
                    linestyle=params["linestyle"],
                    alpha=params.get("alpha", 1),
                )
            else:
                print(f"Simulated Annealing with parameters '{params['label']}' did not find a solution.")

        # plt.axvline(x=ref, color='tab:red', linewidth=4, label='Optimal Energy')

        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Occurrence", fontsize=20)
        plt.title(f"{i} elements", fontsize=20)

        # plt.legend(loc='upper right', frameon=True, fontsize=7)

        plt.savefig(f"SETS2_{i}_SA.png", format="png", bbox_inches="tight")
        plt.close()
        break


run_simulated_annealing_experiment_sets(1)
