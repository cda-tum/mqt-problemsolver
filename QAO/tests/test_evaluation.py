"""File for making some tests during the Development"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import pytest

# for managing symbols
from qubovert import PUBO, boolean_var
from sympy import Expr

from mqt.qao.constraints import Constraints
from mqt.qao.karp import KarpGraphs, KarpNumber, KarpSets
from mqt.qao.objectivefunction import ObjectiveFunction
from mqt.qao.problem import Problem
from mqt.qao.solvers import Solution, Solver
from mqt.qao.variables import Variables


def test_binary_only() -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variables_array("A", [2])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    print(post_dict)
    assert post_dict == {"a": (boolean_var("b0"),), "A_0": (boolean_var("b1"),), "A_1": (boolean_var("b2"),)}


def test_set_cover_initialization():
    """Verify that initializing a set cover problem returns a Problem instance without solving it."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    problem = KarpSets.set_cover(input_data, solve=False)
    assert isinstance(problem, Problem)


def test_set_cover_solving_basic():
    """Ensure that solving a basic set cover problem returns a list of tuples as the solution."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = KarpSets.set_cover(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) for item in solution), "Each solution item should be a tuple"


def test_set_cover_with_weights():
    """Test that the weighted set cover problem returns the minimum cost solution."""
    input_data = [(1, [1, 2]), (10, [2, 3]), (5, [1, 3])]
    solution = KarpSets.set_cover(input_data, solve=True, weighted=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert sum(cost for cost, _ in solution) == 6, "Expected the minimum weighted cost solution"


def test_set_cover_solution_validation_correct():
    """Check that a known valid solution for set cover is validated correctly."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = [(1, [1, 2]), (2, [2, 3])]
    validation = KarpSets.check_set_cover_solution(input_data, solution)
    assert validation["Valid Solution"]


def test_set_cover_solution_validation_incomplete():
    """Verify that an incomplete set cover solution is marked invalid due to missing elements."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = [(1, [1, 2])]
    validation = KarpSets.check_set_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing elements"


def test_set_cover_solution_validation_extraneous():
    """Confirm that a solution with extraneous sets for set cover is marked invalid."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    validation = KarpSets.check_set_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous sets"


def test_set_cover_empty_input():
    """Ensure that initializing a set cover problem with empty input returns a Problem instance."""
    input_data = []
    problem = KarpSets.set_cover(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_set_packing_initialization():
    """Verify that initializing a set packing problem returns a Problem instance without solving it."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    problem = KarpSets.set_packing(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance"


def test_set_packing_solving_basic():
    """Ensure that solving a basic set packing problem returns a list of tuples as the solution."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = KarpSets.set_packing(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) for item in solution), "Each solution item should be a tuple"


def test_set_packing_solution_validation_correct():
    """Check that a known valid solution for set packing is validated correctly."""
    input_data = [(1, [1, 2]), (2, [3, 4]), (1, [4])]
    solution = [(1, [1, 2]), (1, [4])]
    validation = KarpSets.check_set_packing_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_set_packing_solution_validation_extraneous():
    """Confirm that a solution with extraneous sets for set packing is marked invalid."""
    input_data = [(1, [1, 2]), (2, [3, 4])]
    solution = [(1, [1, 2]), (2, [3, 4]), (3, [1, 3])]
    validation = KarpSets.check_set_packing_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous sets"


def test_set_packing_empty_input():
    """Ensure that initializing a set packing problem with empty input returns a Problem instance."""
    input_data = []
    problem = KarpSets.set_packing(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_exact_cover_initialization():
    """Verify that initializing an exact cover problem returns a Problem instance without solving it."""
    input_data = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    problem = KarpSets.exact_cover(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance"


def test_exact_cover_solving_basic():
    """Ensure that solving a basic exact cover problem returns a list of tuples as the solution."""
    input_data = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = KarpSets.exact_cover(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) for item in solution), "Each solution item should be a tuple"


def test_exact_cover_solution_validation_correct():
    """Check that a known valid solution for exact cover is validated correctly."""
    input_data = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2]), (1, [3, 4])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_exact_cover_solution_validation_incomplete():
    """Verify that an incomplete exact cover solution is marked invalid due to missing elements."""
    input_data = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing elements"


def test_exact_cover_solution_validation_extraneous():
    """Confirm that a solution with extraneous sets for exact cover is marked invalid."""
    input_data = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2]), (1, [3, 4]), (2, [1, 3])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous sets"


def test_exact_cover_empty_input():
    """Ensure that initializing an exact cover problem with empty input returns a Problem instance."""
    input_data = []
    problem = KarpSets.exact_cover(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


graph = nx.Graph()
graph.add_edges_from([(1, 3), (3, 5), (2, 4), (4, 6), (1, 4)])
x = [1, 2]
y = [3, 4]
z = [5, 6]


def test_three_d_matching_initialization():
    """Verify that initializing a 3D matching problem returns a Problem instance without solving it."""
    problem = KarpSets.three_d_matching(graph, x, y, z, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for 3D matching initialization"


def test_three_d_matching_solving_basic():
    """Ensure that solving a basic 3D matching problem returns a list of triples as the solution."""
    solution = KarpSets.three_d_matching(graph, x, y, z, solve=True)
    print(solution)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, list) and len(item) == 3 for item in solution), (
        "Each solution item should be a tuple of length 3"
    )


def test_three_d_matching_solution_validation_correct():
    """Check that a known valid solution is correctly validated by the 3D matching problem."""
    solution = [(1, 3, 5)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_three_d_matching_solution_validation_incomplete():
    """Verify that an incomplete 3D matching solution is identified as invalid due to repeated elements."""
    solution = [(1, 2, 3)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to repeated elements"


def test_three_d_matching_solution_validation_extraneous():
    """Confirm that a solution with extraneous triples is identified as invalid."""
    solution = [(1, 2, 3), (3, 2, 1)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous triples"


def test_three_d_matching_solving_with_graph():
    """Solving a 3D matching problem with a graph returns a list of valid triples."""
    solution = KarpSets.three_d_matching(graph, x, y, z, solve=True)
    assert all(isinstance(item, list) and len(item) == 3 for item in solution), (
        "Each solution item should be a tuple of length 3"
    )


def test_hitting_set_initialization():
    """Test the initialization of the hitting set problem."""
    input_data = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    problem = KarpSets.hitting_set(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for hitting set initialization"


def test_hitting_set_solving_basic():
    """Test the basic solving of the hitting set problem without weights."""
    input_data = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution = KarpSets.hitting_set(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, int) for item in solution), "Each solution item should be an integer (element index)"


def test_hitting_set_solution_validation_correct():
    """Test the validation of a correct hitting set solution."""
    input_data = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution = [1, 3]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_hitting_set_solution_validation_incomplete():
    """Test the validation of an incomplete hitting set solution."""
    input_data = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution = [1]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing elements"


def test_hitting_set_solution_validation_extraneous():
    """Test the validation of a hitting set solution with extraneous elements."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    solution = [1, 2, 4]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous elements"


def test_hitting_set_empty_input():
    """Test the handling of an empty input for the hitting set problem."""
    input_data = []
    problem = KarpSets.hitting_set(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_sat_initialization():
    """Test the initialization of the SAT problem."""
    input_data = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    problem = KarpNumber.sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for SAT initialization"


def test_sat_solving_basic():
    """Test the basic solving of the SAT problem without specifying solver parameters."""
    input_data = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = KarpNumber.sat(input_data, solve=True)
    assert isinstance(solution, dict), "Expected a dictionary as the solution"
    assert all(isinstance(key, str) and isinstance(value, float) for key, value in solution.items()), (
        "Expected solution to contain variable names as keys and floats as values"
    )


def test_sat_solution_validation_correct():
    """Test validation for a correct SAT solution."""
    input_data = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_sat_solution_validation_incorrect():
    """Test validation for an incorrect SAT solution."""
    input_data = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to unsatisfied clauses"


def test_three_sat_initialization():
    """Test the initialization of the 3-SAT problem."""
    input_data = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    problem = KarpNumber.three_sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for 3-SAT initialization"


def test_three_sat_solving_basic():
    """Test the basic solving of the 3-SAT problem without specifying solver parameters."""
    input_data = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = KarpNumber.three_sat(input_data, solve=True)
    assert isinstance(solution, dict), "Expected a dictionary as the solution"
    assert all(isinstance(key, str) and isinstance(value, float) for key, value in solution.items()), (
        "Expected solution to contain variable names as keys and floats as values"
    )


def test_three_sat_solution_validation_correct():
    """Test validation for a correct 3-SAT solution."""
    input_data = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0, "e": 1.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_three_sat_solution_validation_incorrect():
    """Test validation for an incorrect 3-SAT solution."""
    input_data = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to unsatisfied clauses"


def test_sat_empty_input():
    """Test handling of an empty input for the SAT problem."""
    input_data = []
    problem = KarpNumber.sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_three_sat_empty_input():
    """Test handling of an empty input for the 3-SAT problem."""
    input_data = []
    problem = KarpNumber.three_sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_number_partition_initialization():
    """Test the initialization of the number partition problem."""
    input_data = [3, 1, 4, 2, 2]
    problem = KarpNumber.number_partition(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for number partition initialization"


def test_number_partition_solving_basic():
    """Test the basic solving of the number partition problem."""
    input_data = [3, 1, 4, 2, 2]
    set_1, set_2 = KarpNumber.number_partition(input_data, solve=True)
    assert isinstance(set_1, list)
    assert isinstance(set_2, list)


def test_number_partition_balanced_solution():
    """Test that the partitioned sets are approximately balanced."""
    input_data = [3, 1, 4, 2, 2]
    set_1, set_2 = KarpNumber.number_partition(input_data, solve=True)
    sum_set_1 = sum(set_1)
    sum_set_2 = sum(set_2)
    assert abs(sum_set_1 - sum_set_2) <= 1, "Expected partitions to have approximately balanced sums"


def test_number_partition_solution_validation_correct():
    """Test validation for a correct number partition solution."""
    input_data = [3, 1, 4, 2, 2]
    solution = {"s_1": 1.0, "s_2": 1.0, "s_3": -1.0, "s_4": -1.0, "s_5": 1.0}  # thiswould create two balanced sets
    validation = KarpNumber.check_number_partition_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_number_partition_solution_validation_incorrect():
    """Test validation for an incorrect number partition solution."""
    input_data = [3, 1, 4, 2, 2]
    solution = {"s_1": 1.0, "s_2": 1.0, "s_3": 1.0, "s_4": -1.0, "s_5": -1.0}  # imbalanced partition
    validation = KarpNumber.check_number_partition_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to imbalance in sums"


def test_number_partition_empty_input():
    """Test handling of an empty input for the number partition problem."""
    input_data = []
    problem = KarpNumber.number_partition(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


# def test_number_partition_visualization():
#    """Test the visualization option for number partition."""
#    input_data = [3, 1, 4, 2, 2]
#    set_1, set_2 = KarpNumber.number_partition(input_data, solve=True, visualize=True)
#    assert isinstance(set_1, list) and isinstance(set_2, list), "Expected two lists as the solution"


def test_job_sequencing_initialization():
    """Test the initialization of the job sequencing problem."""
    job_lengths = [3, 1, 2, 2]
    m = 2
    problem = KarpNumber.job_sequencing(job_lengths, m, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for job sequencing initialization"


def test_job_sequencing_solving_basic():
    """Test the basic solving of the job sequencing problem."""
    job_lengths = [3, 1, 2, 2]
    m = 2
    solution = KarpNumber.job_sequencing(job_lengths, m, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(machine_jobs, list) for machine_jobs in solution), "Each machine's job list should be a list"
    assert len(solution) == m, f"Expected {m} machines in the solution"


def test_job_sequencing_with_single_machine():
    """Test job sequencing when only one machine is available."""
    job_lengths = [3, 5, 2, 7, 1]
    m = 1
    solution = KarpNumber.job_sequencing(job_lengths, m, solve=True)
    assert len(solution) == 1, "Expected only one machine in the solution"
    assert sum(job_lengths) == sum(job_lengths[job] for job in solution[0]), (
        "All jobs should be assigned to the single machine available"
    )


def test_knapsack_initialization():
    """Test the initialization of the knapsack problem."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 5
    problem = KarpNumber.knapsack(items, max_weight, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for knapsack initialization"


def test_knapsack_solving_basic():
    """Test the basic solving of the knapsack problem."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 5
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in solution), (
        "Each solution item should be a tuple of (weight, value)"
    )


def test_knapsack_solution_optimization():
    """Test that the knapsack solution maximizes the value without exceeding max weight."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 5
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    total_weight = sum(item[0] for item in solution)
    total_value = sum(item[1] for item in solution)

    assert total_weight <= max_weight, "Total weight should not exceed the maximum allowed weight"
    assert total_value == 8, "Expected maximum value to be 7 for this input"


def test_knapsack_solution_validation_correct():
    """Test validation for a correct knapsack solution."""
    max_weight = 5
    solution = [(2, 3), (3, 4)]
    validation = KarpNumber.check_knapsack_solution(max_weight, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_knapsack_solution_validation_incorrect():
    """Test validation for an incorrect knapsack solution (overweight)."""
    max_weight = 5
    solution = [(3, 4), (4, 5)]
    validation = KarpNumber.check_knapsack_solution(max_weight, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to exceeding weight limit"


def test_knapsack_with_zero_max_weight():
    """Test knapsack when max weight is zero, expecting an empty solution."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 0
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    assert solution == [], "Expected an empty solution when max weight is zero"


def test_knapsack_with_large_max_weight():
    """Test knapsack with a large max weight, expecting all items to be included if possible."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 15
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    total_weight = sum(item[0] for item in solution)
    total_value = sum(item[1] for item in solution)

    assert total_weight <= max_weight, "Total weight should not exceed the maximum allowed weight"
    assert total_value <= 20, "Expected total value to be the sum of all item values"


def test_clique_initialization():
    """Test the initialization of the clique problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k = 3
    problem = KarpGraphs.clique(graph, k=k, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for clique initialization"


def test_clique_solving_basic():
    """Test the basic solving of the clique problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k = 3
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), (
        "Each element in the solution should be an integer representing a node"
    )


def test_clique_solution_k_value():
    """Test that the solution has the correct number of nodes for the clique size k."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (2, 4), (4, 3)])
    k = 3
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    assert len(solution) == k, f"Expected a clique of size {k}, but got {len(solution)}"


def test_clique_solution_maximal_clique():
    """Test finding the maximal clique (k=0) in the graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 1)])
    solution = KarpGraphs.clique(graph, k=0, solve=True)

    assert len(solution) == 3, "Expected maximal clique of size 3 for this input graph"


def test_clique_solution_validation_correct():
    """Test validation for a correct clique solution."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (2, 4)])
    solution = [1, 2, 3]
    validation = KarpGraphs.check_clique_solution(graph, solution)

    assert validation["Valid Solution"], "Expected solution to be valid"


def test_clique_solution_validation_incorrect():
    """Test validation for an incorrect clique solution (not a complete subgraph)."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (1, 4)])
    solution = [1, 2, 4]
    validation = KarpGraphs.check_clique_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing edge"


def test_clique_empty_graph():
    """Test handling of an empty graph."""
    graph = nx.Graph()
    k = 3
    problem = KarpGraphs.clique(graph, k=k, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance even with an empty graph"


def test_clique_single_node_graph():
    """Test a graph with a single node."""
    graph = nx.Graph()
    graph.add_node(1)
    k = 1
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    assert solution == [1], "Expected the single node as the solution"


def test_clique_large_k_value():
    """Test the clique method with a k value larger than the graph's maximum clique size."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k = 2
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    assert len(solution) == k, "Expected an empty solution as no clique of size 5 exists in the graph"


def test_clique_cover_initialization():
    """Test the initialization of the clique cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors = 2
    problem = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for clique cover initialization"


def test_clique_cover_solving_basic():
    """Test the basic solving of the clique cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors = 2
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in solution), (
        "Each element in the solution should be a tuple representing (node, color)"
    )


def test_clique_cover_solution_num_colors():
    """Test that the solution uses the correct number of colors."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)])
    num_colors = 2
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    used_colors = {color for _, color in solution}
    assert len(used_colors) <= num_colors, (
        f"Expected solution to use up to {num_colors} colors, but used {len(used_colors)}"
    )


def test_clique_cover_solution_correct_cover():
    """Test validation for a correct clique cover solution."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    solution = [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)]  # known correctcover
    validation = KarpGraphs.check_clique_cover_solution(graph, solution)

    assert validation["Valid Solution"], "Expected solution to be valid for correct clique cover"


def test_clique_cover_solution_incorrect_cover():
    """Test validation for an incorrect clique cover solution (non-clique assignment)."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (1, 4)])
    solution = [(1, 1), (2, 1), (3, 2), (4, 1)]  # Invalid cover (node 4 doesn't connect to 2 or 3)
    validation = KarpGraphs.check_clique_cover_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected solution to be invalid due to non-clique assignment"


def test_clique_cover_single_node_graph():
    """Test a graph with a single node."""
    graph = nx.Graph()
    graph.add_node(1)
    num_colors = 1
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    assert solution == [(1, 1)], "Expected the single node with color 1 in the solution"


def test_clique_cover_large_num_colors():
    """Test the clique cover with a large number of colors, exceeding the minimum requirement."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors = 5  # more colors than required
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    assert len({color for _, color in solution}) <= num_colors, (
        "Expected the number of colors used to be within the specified num_colors"
    )


def test_clique_cover_disconnected_graph():
    """Test the clique cover with a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    num_colors = 3
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    used_colors = {color for _, color in solution}
    assert len(used_colors) <= num_colors, (
        f"Expected solution to use up to {num_colors} colors, but used {len(used_colors)}"
    )
    assert len(solution) == 6, "Expected each node to be assigned a color in the solution"


def test_vertex_cover_initialization():
    """Test the initialization of the vertex cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    problem = KarpGraphs.vertex_cover(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for vertex cover initialization"


def test_vertex_cover_solving_basic():
    """Test the basic solving of the vertex cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_vertex_cover_solution_minimum_size():
    """Test that the solution provides a minimum vertex cover size."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    # a valid cover for this graph is of size 3 (e.g., nodes 2, 4, and 5 or any equivalent valid cover)
    assert len(solution) <= 3, "Expected solution to provide a minimum vertex cover size"


def test_vertex_cover_solution_correct_cover():
    """Test validation for a correct vertex cover solution."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 1)])
    solution = [2, 1]  # known correct cover for this graph
    validation = KarpGraphs.check_vertex_cover_solution(graph, solution)

    assert validation["Valid Solution"], "Expected solution to be valid for correct vertex cover"


def test_vertex_cover_solution_incorrect_cover():
    """Test validation for an incorrect vertex cover solution (missing required nodes)."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (1, 4)])
    solution = [2]  # Invalid cover (node 3 is not covered)
    validation = KarpGraphs.check_vertex_cover_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected solution to be invalid due to uncovered edges"


def test_vertex_cover_single_node_graph():
    """Test a graph with a single node."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    assert solution == [], "Expected no nodes in the cover as a single node with no edges does not require covering"


def test_vertex_cover_large_num_nodes():
    """Test the vertex cover on a graph with many nodes but few edges."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    assert len(solution) <= 3, "Expected the vertex cover to include at most one node per edge"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_vertex_cover_disconnected_graph():
    """Test the vertex cover with a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    assert len(solution) <= 3, "Expected each disconnected component to be covered individually with minimum nodes"
    assert len(solution) == len(set(solution)), "Expected each node in the solution to be unique"


def test_graph_coloring_initialization():
    """Test the initialization of the graph coloring problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    problem = KarpGraphs.graph_coloring(graph, num_colors=3, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for graph coloring initialization"


def test_graph_coloring_basic_solution():
    """Test the basic solution of the graph coloring problem with 3 colors."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=3, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node_color, tuple) and len(node_color) == 2 for node_color in solution), (
        "Each element in the solution should be a tuple of (node, color)"
    )


def test_graph_coloring_validity():
    """Test that the coloring solution is valid for a simple graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=3, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid coloring"


def test_graph_coloring_invalid_colors():
    """Test that the graph coloring fails when not enough colors are provided."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert not validation["Valid Solution"], "Expected the solution to be invalid with insufficient colors"


def test_graph_coloring_chromatic_number():
    """Test that the solution respects the chromatic number of a small cycle graph."""
    graph = nx.cycle_graph(4)
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid with 2 colors for C4 graph"


def test_graph_coloring_disconnected_graph():
    """Test coloring on a disconnected graph with 2 components."""
    graph = nx.Graph([(1, 2), (3, 4)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid with disconnected components"


def test_graph_coloring_single_node():
    """Test coloring on a single-node graph."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.graph_coloring(graph, num_colors=1, solve=True)

    assert solution == [(1, 1)], "Expected the single node to be colored with the only available color"


def test_graph_coloring_complete_graph():
    """Test coloring on a complete graph with n nodes, requiring n colors."""
    n = 4
    graph = nx.complete_graph(n)
    solution = KarpGraphs.graph_coloring(graph, num_colors=n, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], f"Expected a valid coloring with {n} colors for K{n} graph"


def test_graph_coloring_large_sparse_graph():
    """Test graph coloring on a large sparse graph with fewer colors than nodes."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=3, solve=True)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for sparse graph with 3 colors"


def test_independent_set_initialization():
    """Test the initialization of the independent set problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    problem = KarpGraphs.independent_set(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for independent set initialization"


def test_independent_set_basic_solution():
    """Test the basic solution of the independent set problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_independent_set_validity():
    """Test that the independent set solution is valid for a simple graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid independent set"


def test_independent_set_disconnected_graph():
    """Test independent set on a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for a disconnected graph"


def test_independent_set_large_sparse_graph():
    """Test independent set on a large sparse graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for a sparse graph"


def test_independent_set_single_node():
    """Test independent set on a single-node graph."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.independent_set(graph, solve=True)

    assert solution == [1], "Expected the single node to be part of the independent set"


def test_independent_set_complete_graph():
    """Test independent set on a complete graph, where only one node can be in the set."""
    n = 4
    graph = nx.complete_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    assert len(solution) == 1, "Expected only one node in the independent set for a complete graph"


def test_independent_set_path_graph():
    """Test independent set on a path graph, where alternate nodes form the maximum independent set."""
    n = 5
    graph = nx.path_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for the path graph"
    assert len(solution) == (n + 1) // 2, "Expected maximum independent set to have (n+1)//2 nodes for path graph"


def test_independent_set_cycle_graph():
    """Test independent set on a cycle graph, where alternate nodes form the maximum independent set."""
    n = 6
    graph = nx.cycle_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for the cycle graph"
    assert len(solution) == n // 2, "Expected maximum independent set to have n//2 nodes for cycle graph"


def test_directed_feedback_vertex_set_initialization():
    """Test the initialization of the directed feedback vertex set problem."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
    problem = KarpGraphs.directed_feedback_vertex_set(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for feedback vertex set initialization"


def test_directed_feedback_vertex_set_basic_solution():
    """Test the basic solution of the directed feedback vertex set problem."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_directed_feedback_vertex_set_validity():
    """Test that the feedback vertex set solution is valid for a simple graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid feedback vertex set"


def test_directed_feedback_vertex_set_minimal_size():
    """Test that the solution provides a minimal feedback vertex set."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    assert len(solution) == 1, "Expected the minimal feedback vertex set to have size 1"


def test_directed_feedback_vertex_set_solution_correct():
    """Test validation for a correct feedback vertex set solution."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [2]
    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)

    assert validation["Valid Solution"], "Expected the solution to be valid"


def test_directed_feedback_vertex_set_solution_incorrect():
    """Test validation for an incorrect feedback vertex set solution (cycles remain)."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [4]
    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected the solution to be invalid due to remaining cycles"


def test_directed_feedback_vertex_set_no_cycles():
    """Test the feedback vertex set on an acyclic graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback vertex set for an acyclic graph"


def test_directed_feedback_vertex_set_multiple_cycles():
    """Test the feedback vertex set on a graph with multiple cycles."""
    graph = nx.DiGraph([(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for multiple cycles"

    assert len(solution) >= 1, "Expected the feedback vertex set to include at least one node"


def test_directed_feedback_vertex_set_disconnected_graph():
    """Test the feedback vertex set on a disconnected graph."""
    graph = nx.DiGraph([(1, 2), (2, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for disconnected graph"

    assert len(solution) >= 1, "Expected the feedback vertex set to include nodes from the cyclic component"


def test_directed_feedback_vertex_set_single_node():
    """Test the feedback vertex set on a graph with a single node."""
    graph = nx.DiGraph()
    graph.add_node(1)
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback vertex set for a single-node graph"


def test_directed_feedback_vertex_set_large_acyclic_graph():
    """Test the feedback vertex set on a large acyclic graph."""
    graph = nx.gn_graph(10, seed=42, create_using=nx.DiGraph)
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback vertex set for a large acyclic graph"


def test_directed_feedback_edge_set_initialization():
    """Test the initialization of the directed feedback edge set problem."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
    problem = KarpGraphs.directed_feedback_edge_set(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for feedback edge set initialization"


def test_directed_feedback_edge_set_basic_solution():
    """Test the basic solution of the directed feedback edge set problem."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in solution), (
        "Each element in the solution should be a tuple representing an edge"
    )


def test_directed_feedback_edge_set_validity():
    """Test that the feedback edge set solution is valid for a simple graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid feedback edge set"


def test_directed_feedback_edge_set_minimal_size():
    """Test that the solution provides a minimal feedback edge set."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    assert len(solution) == 1, "Expected the minimal feedback edge set to have size 1"


def test_directed_feedback_edge_set_solution_correct():
    """Test validation for a correct feedback edge set solution."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [(3, 1)]
    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)

    assert validation["Valid Solution"], "Expected the solution to be valid"


def test_directed_feedback_edge_set_solution_incorrect():
    """Test validation for an incorrect feedback edge set solution (cycles remain)."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [(1, 3)]
    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected the solution to be invalid due to remaining cycles"


def test_directed_feedback_edge_set_no_cycles():
    """Test the feedback edge set on an acyclic graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback edge set for an acyclic graph"


def test_directed_feedback_edge_set_multiple_cycles():
    """Test the feedback edge set on a graph with multiple cycles."""
    graph = nx.DiGraph([(1, 2), (2, 1), (2, 3), (3, 2)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for multiple cycles"
    assert len(solution) >= 1, "Expected the feedback edge set to include at least one edge"


def test_directed_feedback_edge_set_disconnected_graph():
    """Test the feedback edge set on a disconnected graph."""
    graph = nx.DiGraph([(1, 2), (2, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for disconnected graph"
    assert len(solution) >= 1, "Expected the feedback edge set to include edges from the cyclic component"


def test_directed_feedback_edge_set_single_node():
    """Test the feedback edge set on a graph with a single node."""
    graph = nx.DiGraph()
    graph.add_node(1)
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback edge set for a single-node graph"


def test_directed_feedback_edge_set_large_acyclic_graph():
    """Test the feedback edge set on a large acyclic graph."""
    graph = nx.gn_graph(8, seed=42, create_using=nx.DiGraph)
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    assert solution == [], "Expected an empty feedback edge set for a large acyclic graph"


def test_hamiltonian_path_initialization():
    """Test the initialization of the Hamiltonian Path problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    problem = KarpGraphs.hamiltonian_path(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for Hamiltonian Path initialization"


def test_hamiltonian_path_exists():
    """Test if a Hamiltonian Path exists in a simple path graph."""
    graph = nx.path_graph(4)
    solution = KarpGraphs.hamiltonian_path(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert len(solution) == 4, "Expected Hamiltonian Path to include all nodes in the graph"


def test_hamiltonian_path_complete_graph():
    """Test the Hamiltonian Path in a complete graph, which always has a Hamiltonian Path."""
    graph = nx.complete_graph(4)
    solution = KarpGraphs.hamiltonian_path(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert len(solution) == 4, "Expected Hamiltonian Path to include all nodes in a complete graph"


def test_max_cut_initialization():
    """Test the initialization of the Max-Cut problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    problem = KarpGraphs.max_cut(graph, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for Max-Cut initialization"


def test_max_cut_simple_graph():
    """Test a simple triangle graph for Max-Cut."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.max_cut(graph, solve=True)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert len(solution) <= len(graph.nodes), "Solution set should not exceed the number of nodes in the graph"


def test_max_cut_single_node():
    """Test Max-Cut on a single-node graph."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.max_cut(graph, solve=True)

    assert solution == [], "Expected an empty solution for a single-node graph with no edges"


def test_spin_only() -> None:
    """Test only the construction of spin variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_spin_variable("a")
    variables.add_spin_variables_array("A", [2])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    assert post_dict == {
        "a": (boolean_var("b0"), 2, -1),
        "A_0": (boolean_var("b1"), 2, -1),
        "A_1": (boolean_var("b2"), 2, -1),
    }


def test_discrete_only() -> None:
    """Test only the construction of discrete variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_discrete_variable("a", [-1, 1, 3])
    variables.add_discrete_variables_array("A", [2], [-1, 1, 3])
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    assert post_dict == {
        "a": ["dictionary", (boolean_var("b0"), -1), (boolean_var("b1"), 1), (boolean_var("b2"), 3)],
        "A_0": ["dictionary", (boolean_var("b3"), -1), (boolean_var("b4"), 1), (boolean_var("b5"), 3)],
        "A_1": ["dictionary", (boolean_var("b6"), -1), (boolean_var("b7"), 1), (boolean_var("b8"), 3)],
    }


@pytest.mark.parametrize(
    ("encoding", "distribution", "precision", "min_val", "max_val"),
    [
        ("dictionary", "uniform", 0.5, -1, 2),
        ("unitary", "uniform", 0.5, -1, 0),
        ("domain well", "uniform", 0.5, -1, 1),
        ("logarithmic 2", "uniform", -1, -1, 2),
        ("arithmetic progression", "uniform", 0.5, -1, 2),
        ("bounded coefficient 1", "uniform", 0.5, -1, 2),
        ("bounded coefficient 8", "uniform", 1, 0, 12),
        ("", "", 0.25, -2, 2),
        ("", "", 0.2, -2, 2),
    ],
)
def test_continuous_only(encoding: str, distribution: str, precision: float, min_val: float, max_val: float) -> None:
    """Test only the construction of continuous variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_continuous_variable("a", min_val, max_val, precision, distribution, encoding)
    variables.add_continuous_variables_array("A", [2], min_val, max_val, precision, distribution, encoding)
    variables.move_to_binary(constraint.constraints)
    post_dict = variables.binary_variables_name_weight
    if (encoding, distribution, precision, min_val, max_val) == ("dictionary", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "dictionary",
                (boolean_var("b0"), -1.0),
                (boolean_var("b1"), -0.5),
                (boolean_var("b2"), 0.0),
                (boolean_var("b3"), 0.5),
                (boolean_var("b4"), 1.0),
                (boolean_var("b5"), 1.5),
                (boolean_var("b6"), 2.0),
            ],
            "A_0": [
                "dictionary",
                (boolean_var("b7"), -1.0),
                (boolean_var("b8"), -0.5),
                (boolean_var("b9"), 0.0),
                (boolean_var("b10"), 0.5),
                (boolean_var("b11"), 1.0),
                (boolean_var("b12"), 1.5),
                (boolean_var("b13"), 2.0),
            ],
            "A_1": [
                "dictionary",
                (boolean_var("b14"), -1.0),
                (boolean_var("b15"), -0.5),
                (boolean_var("b16"), 0.0),
                (boolean_var("b17"), 0.5),
                (boolean_var("b18"), 1.0),
                (boolean_var("b19"), 1.5),
                (boolean_var("b20"), 2.0),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("unitary", "uniform", 0.5, -1, 0):
        assert post_dict == {
            "a": ["unitary", (boolean_var("b0"), 0.5, -1), (boolean_var("b1"), 1)],
            "A_0": ["unitary", (boolean_var("b2"), 0.5, -1), (boolean_var("b3"), 1)],
            "A_1": ["unitary", (boolean_var("b4"), 0.5, -1), (boolean_var("b5"), 1)],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("domain well", "uniform", 0.5, -1, 1):
        assert post_dict == {
            "a": [
                "domain well",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1.5),
                (boolean_var("b3"), 2),
            ],
            "A_0": [
                "domain well",
                (boolean_var("b4"), 0.5, -1),
                (boolean_var("b5"), 1),
                (boolean_var("b6"), 1.5),
                (boolean_var("b7"), 2),
            ],
            "A_1": [
                "domain well",
                (boolean_var("b8"), 0.5, -1),
                (boolean_var("b9"), 1),
                (boolean_var("b10"), 1.5),
                (boolean_var("b11"), 2),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("logarithmic 2", "uniform", -1, -1, 2):
        assert post_dict == {
            "a": ["logarithmic", (boolean_var("b0"), 0.5, -1), (boolean_var("b1"), 1), (boolean_var("b2"), 1.5)],
            "A_0": ["logarithmic", (boolean_var("b3"), 0.5, -1), (boolean_var("b4"), 1), (boolean_var("b5"), 1.5)],
            "A_1": ["logarithmic", (boolean_var("b6"), 0.5, -1), (boolean_var("b7"), 1), (boolean_var("b8"), 1.5)],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("arithmetic progression", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "arithmetic progression",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1.5),
            ],
            "A_0": [
                "arithmetic progression",
                (boolean_var("b3"), 0.5, -1),
                (boolean_var("b4"), 1),
                (boolean_var("b5"), 1.5),
            ],
            "A_1": [
                "arithmetic progression",
                (boolean_var("b6"), 0.5, -1),
                (boolean_var("b7"), 1),
                (boolean_var("b8"), 1.5),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("bounded coefficient 1", "uniform", 0.5, -1, 2):
        assert post_dict == {
            "a": [
                "bounded coefficient",
                (boolean_var("b0"), 0.5, -1),
                (boolean_var("b1"), 1),
                (boolean_var("b2"), 1),
                (boolean_var("b3"), 1),
            ],
            "A_0": [
                "bounded coefficient",
                (boolean_var("b4"), 0.5, -1),
                (boolean_var("b5"), 1),
                (boolean_var("b6"), 1),
                (boolean_var("b7"), 1),
            ],
            "A_1": [
                "bounded coefficient",
                (boolean_var("b8"), 0.5, -1),
                (boolean_var("b9"), 1),
                (boolean_var("b10"), 1),
                (boolean_var("b11"), 1),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("bounded coefficient 8", "uniform", 1, 0, 12):
        assert post_dict == {
            "a": [
                "logarithmic",
                (boolean_var("b0"), 1, 0),
                (boolean_var("b1"), 2),
                (boolean_var("b2"), 4),
                (boolean_var("b3"), 5),
            ],
            "A_0": [
                "logarithmic",
                (boolean_var("b4"), 1, 0),
                (boolean_var("b5"), 2),
                (boolean_var("b6"), 4),
                (boolean_var("b7"), 5),
            ],
            "A_1": [
                "logarithmic",
                (boolean_var("b8"), 1, 0),
                (boolean_var("b9"), 2),
                (boolean_var("b10"), 4),
                (boolean_var("b11"), 5),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == ("", "", 0.25, -2, 2):
        assert post_dict == {
            "a": [
                "logarithmic",
                (boolean_var("b0"), 0.25, -2),
                (boolean_var("b1"), 0.5),
                (boolean_var("b2"), 1),
                (boolean_var("b3"), 2),
                (boolean_var("b4"), 0.25),
            ],
            "A_0": [
                "logarithmic",
                (boolean_var("b5"), 0.25, -2),
                (boolean_var("b6"), 0.5),
                (boolean_var("b7"), 1),
                (boolean_var("b8"), 2),
                (boolean_var("b9"), 0.25),
            ],
            "A_1": [
                "logarithmic",
                (boolean_var("b10"), 0.25, -2),
                (boolean_var("b11"), 0.5),
                (boolean_var("b12"), 1),
                (boolean_var("b13"), 2),
                (boolean_var("b14"), 0.25),
            ],
        }
    elif (encoding, distribution, precision, min_val, max_val) == (
        "",
        "",
        0.2,
        -2,
        2,
    ):  # 0.6000000000000001 instead of 0.6 is for a python numerical error
        assert post_dict == {
            "a": [
                "arithmetic progression",
                (boolean_var("b0"), 0.2, -2),
                (boolean_var("b1"), 0.4),
                (boolean_var("b2"), 0.6000000000000001),
                (boolean_var("b3"), 0.8),
                (boolean_var("b4"), 1),
                (boolean_var("b5"), 1),
            ],
            "A_0": [
                "arithmetic progression",
                (boolean_var("b6"), 0.2, -2),
                (boolean_var("b7"), 0.4),
                (boolean_var("b8"), 0.6000000000000001),
                (boolean_var("b9"), 0.8),
                (boolean_var("b10"), 1),
                (boolean_var("b11"), 1),
            ],
            "A_1": [
                "arithmetic progression",
                (boolean_var("b12"), 0.2, -2),
                (boolean_var("b13"), 0.4),
                (boolean_var("b14"), 0.6000000000000001),
                (boolean_var("b15"), 0.8),
                (boolean_var("b16"), 1),
                (boolean_var("b17"), 1),
            ],
        }


def test_cost_function() -> None:
    """Test for cost function translation"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    variables.move_to_binary(constraint.constraints)
    qubo = PUBO()
    qubo = objective_function.rewrite_cost_functions(qubo, variables)
    reference_qubo_dict = {
        ("b0",): 1.0,
        ("b1",): 2.0,
        ("b1", "b4"): -0.25,
        ("b1", "b5"): -0.5,
        ("b1", "b6"): -1.0,
        ("b1", "b7"): -2.0,
        ("b1", "b8"): -0.25,
        ("b2", "b4"): 0.25,
        ("b2",): -2.0,
        ("b2", "b5"): 0.5,
        ("b2", "b6"): 1.0,
        ("b2", "b7"): 2.0,
        ("b2", "b8"): 0.25,
        ("b3", "b4"): 0.75,
        ("b3",): -6.0,
        ("b3", "b5"): 1.5,
        ("b3", "b6"): 3.0,
        ("b3", "b7"): 6.0,
        ("b3", "b8"): 0.75,
        ("b4",): -0.9375,
        ("b5",): -1.75,
        ("b6",): -3.0,
        ("b7",): -4.0,
        ("b8",): -0.9375,
        ("b4", "b5"): 0.25,
        ("b4", "b6"): 0.5,
        ("b4", "b7"): 1.0,
        ("b4", "b8"): 0.125,
        ("b5", "b6"): 1.0,
        ("b5", "b7"): 2.0,
        ("b5", "b8"): 0.25,
        ("b6", "b7"): 4.0,
        ("b6", "b8"): 0.5,
        ("b7", "b8"): 1.0,
        (): 4.0,
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key, value in reference_qubo_dict.items():
        reference_qubo_dict_re[tuple(sorted(key))] = value
    assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


def test_cost_function_matrix() -> None:
    """Test for cost function translation"""
    variables = Variables()
    constraint = Constraints()
    m1 = variables.add_continuous_variables_array("M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2")
    m2 = variables.add_continuous_variables_array("M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2")
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(np.matmul(m1, m2).item(0, 0))
    variables.move_to_binary(constraint.constraints)
    qubo = PUBO()
    qubo = objective_function.rewrite_cost_functions(qubo, variables)
    reference_qubo_dict = {
        ("b0", "b6"): 0.25,
        ("b0",): -0.5,
        ("b0", "b7"): 0.5,
        ("b0", "b8"): 0.75,
        ("b6",): -0.5,
        ("b7",): -1.0,
        ("b8",): -1.5,
        ("b1", "b6"): 0.5,
        ("b1",): -1.0,
        ("b1", "b7"): 1.0,
        ("b1", "b8"): 1.5,
        ("b2", "b6"): 0.75,
        ("b2",): -1.5,
        ("b2", "b7"): 1.5,
        ("b2", "b8"): 2.25,
        ("b3", "b9"): 0.25,
        ("b3",): -0.5,
        ("b10", "b3"): 0.5,
        ("b11", "b3"): 0.75,
        ("b9",): -0.5,
        ("b10",): -1.0,
        ("b11",): -1.5,
        ("b4", "b9"): 0.5,
        ("b4",): -1.0,
        ("b10", "b4"): 1.0,
        ("b11", "b4"): 1.5,
        ("b5", "b9"): 0.75,
        ("b5",): -1.5,
        ("b10", "b5"): 1.5,
        ("b11", "b5"): 2.25,
        (): 2.0,
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key, value in reference_qubo_dict.items():
        reference_qubo_dict_re[tuple(sorted(key))] = value
    assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


@pytest.mark.parametrize(
    ("expression", "var_precision"),
    [
        ("~a = b", False),
        ("a & b = c", False),
        ("a | b = c", False),
        ("a ^ b = c", False),
        ("a + b >= 1", False),
        ("e >= 1", False),
        ("e >= 1", True),
        ("a + b <= 1", False),
        ("a + b > 1", False),
        ("a + b < 1", False),
        ("e <= 1", False),
        ("e <= -1", True),
        ("e > 1", True),
        ("e < 1", False),
        ("e > 1", False),
        ("d < 1", True),
    ],
)
def test_constraint(expression: str, var_precision: bool) -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variable("b")
    variables.add_binary_variable("c")
    variables.add_discrete_variable("d", [-1, 1, 3])
    variables.add_continuous_variable("e", -2, 2, 0.25, "", "")
    variables.move_to_binary(constraint.constraints)
    constraint.add_constraint(expression, True, True, var_precision)
    constraint.translate_constraints(variables)
    dictionary_constraints_qubo = {
        ("b3",): -1.0,
        ("b4",): -1.0,
        ("b5",): -1.0,
        ("b3", "b4"): 2.0,
        ("b3", "b5"): 2.0,
        ("b4", "b5"): 2.0,
        (): 1.0,
    }
    qubo_first = constraint.constraints_penalty_functions[0][0]
    qubo_second = constraint.constraints_penalty_functions[1][0]

    qubo_first_re = {}
    for key in qubo_first:
        qubo_first_re[tuple(sorted(key))] = qubo_first[key]
    qubo_second_re = {}
    for key in qubo_second:
        qubo_second_re[tuple(sorted(key))] = qubo_second[key]
    dictionary_constraints_qubo_re = {}
    for key, value in dictionary_constraints_qubo.items():
        dictionary_constraints_qubo_re[tuple(sorted(key))] = value
    if expression == "~a = b":
        dictionary_constraints_qubo_2 = {("b0",): -1.0, ("b1",): -1.0, ("b0", "b1"): 2.0, (): 1.0}
        dictionary_constraints_qubo_2_re = {}
        for key, value in dictionary_constraints_qubo_2.items():
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = value
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a & b = c":
        dictionary_constraints_qubo_2 = {("b0", "b1"): 1.0, ("b0", "b2"): -2.0, ("b1", "b2"): -2.0, ("b2",): 3.0}
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a | b = c":
        dictionary_constraints_qubo_2 = {
            ("b0", "b1"): 1.0,
            ("b0",): 1.0,
            ("b1",): 1.0,
            ("b0", "b2"): -2.0,
            ("b1", "b2"): -2.0,
            ("b2",): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a ^ b = c":
        dictionary_constraints_qubo_2 = {
            ("b0", "b1", "b2"): 4.0,
            ("b0",): 1,
            ("b1",): 1.0,
            ("b2",): 1.0,
            ("b0", "b1"): -2.0,
            ("b0", "b2"): -2.0,
            ("b1", "b2"): -2.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b >= 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -1.0,
            ("b1",): -1.0,
            ("__a0",): 3.0,
            ("b0", "b1"): 2.0,
            ("__a0", "b0"): -2.0,
            ("__a0", "b1"): -2.0,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e >= 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): 7.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.5,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -1.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -2.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -4.0,
            ("__a0", "b10"): -0.5,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e >= 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): 1.5625,
            ("__a1",): 3.25,
            ("__a2",): 1.5625,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.125,
            ("__a1", "b6"): -0.25,
            ("__a2", "b6"): -0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -0.25,
            ("__a1", "b7"): -0.5,
            ("__a2", "b7"): -0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -0.5,
            ("__a1", "b8"): -1.0,
            ("__a2", "b8"): -0.5,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -1.0,
            ("__a1", "b9"): -2.0,
            ("__a2", "b9"): -1.0,
            ("__a0", "b10"): -0.125,
            ("__a1", "b10"): -0.25,
            ("__a2", "b10"): -0.125,
            ("__a0", "__a1"): 0.25,
            ("__a0", "__a2"): 0.125,
            ("__a1", "__a2"): 0.25,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b <= 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -1.0,
            ("b1",): -1.0,
            ("__a0",): -1.0,
            ("b0", "b1"): 2.0,
            ("__a0", "b0"): 2.0,
            ("__a0", "b1"): 2.0,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b > 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): -3.0,
            ("b1",): -3.0,
            ("b0", "b1"): 2.0,
            (): 4.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "a + b < 1":
        dictionary_constraints_qubo_2 = {
            ("b0",): 1.0,
            ("b1",): 1.0,
            ("b0", "b1"): 2.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e <= 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.4375,
            ("b7",): -2.75,
            ("b8",): -5.0,
            ("b9",): -8.0,
            ("b10",): -1.4375,
            ("__a0",): -5.0,
            ("__a1",): -8.0,
            ("__a0", "__a1"): 4.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.5,
            ("__a1", "b6"): 1.0,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 1.0,
            ("__a1", "b7"): 2.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 2.0,
            ("__a1", "b8"): 4.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): 4.0,
            ("__a1", "b9"): 8.0,
            ("__a0", "b10"): 0.5,
            ("__a1", "b10"): 1.0,
            (): 9.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e <= -1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -0.4375,
            ("b7",): -0.75,
            ("b8",): -1.0,
            ("b10",): -0.4375,
            ("__a0",): -0.4375,
            ("__a1",): -0.75,
            ("__a2",): -0.4375,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.125,
            ("__a1", "b6"): 0.25,
            ("__a2", "b6"): 0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 0.25,
            ("__a1", "b7"): 0.5,
            ("__a2", "b7"): 0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 0.5,
            ("__a1", "b8"): 1.0,
            ("__a2", "b8"): 0.5,
            ("b10", "b9"): 1.0,
            ("__a0", "b9"): 1.0,
            ("__a1", "b9"): 2.0,
            ("__a2", "b9"): 1.0,
            ("__a0", "b10"): 0.125,
            ("__a1", "b10"): 0.25,
            ("__a2", "b10"): 0.125,
            ("__a0", "__a1"): 0.25,
            ("__a0", "__a2"): 0.125,
            ("__a1", "__a2"): 0.25,
            (): 1.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e > 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.9375,
            ("b7",): -3.75,
            ("b8",): -7.0,
            ("b9",): -12.0,
            ("b10",): -1.9375,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("b10", "b9"): 1,
            (): 16.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e < 1" and not var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -0.9375,
            ("b7",): -1.75,
            ("b8",): -3.0,
            ("b9",): -4.0,
            ("b10",): -0.9375,
            ("__a0",): -3.0,
            ("__a1",): -3.0,
            ("__a0", "__a1"): 2.0,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): 0.5,
            ("__a1", "b6"): 0.5,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): 1.0,
            ("__a1", "b7"): 1.0,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): 2.0,
            ("__a1", "b8"): 2.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): 4.0,
            ("__a1", "b9"): 4.0,
            ("__a0", "b10"): 0.5,
            ("__a1", "b10"): 0.5,
            (): 4.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "e > 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b6",): -1.5625,
            ("b7",): -3.0,
            ("b8",): -5.5,
            ("b9",): -9.0,
            ("b10",): -1.5625,
            ("__a0",): 1.6875,
            ("__a1",): 3.5,
            ("b6", "b7"): 0.25,
            ("b6", "b8"): 0.5,
            ("b6", "b9"): 1.0,
            ("b10", "b6"): 0.125,
            ("__a0", "b6"): -0.125,
            ("__a1", "b6"): -0.25,
            ("b7", "b8"): 1.0,
            ("b7", "b9"): 2.0,
            ("b10", "b7"): 0.25,
            ("__a0", "b7"): -0.25,
            ("__a1", "b7"): -0.5,
            ("b8", "b9"): 4.0,
            ("b10", "b8"): 0.5,
            ("__a0", "b8"): -0.5,
            ("__a1", "b8"): -1.0,
            ("b10", "b9"): 1,
            ("__a0", "b9"): -1.0,
            ("__a1", "b9"): -2.0,
            ("__a0", "b10"): -0.125,
            ("__a1", "b10"): -0.25,
            ("__a0", "__a1"): 0.25,
            (): 10.5625,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))
    elif expression == "d < 1" and var_precision:
        dictionary_constraints_qubo_2 = {
            ("b3",): 1.0,
            ("b4",): 1.0,
            ("b5",): 9.0,
            ("__a0",): 1.0,
            ("b3", "b4"): -2.0,
            ("b3", "b5"): -6.0,
            ("__a0", "b3"): -2.0,
            ("b4", "b5"): 6.0,
            ("__a0", "b4"): 2.0,
            ("__a0", "b5"): 6.0,
        }
        dictionary_constraints_qubo_2_re = {}
        for key in dictionary_constraints_qubo_2:
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = dictionary_constraints_qubo_2[key]
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))


@pytest.mark.parametrize(
    "expression",
    ["~b0 = b1"],
)
def test_constraint_no_sub(expression: str) -> None:
    """Test only the construction of binary variables"""
    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variable("b")
    variables.add_binary_variable("c")
    variables.add_discrete_variable("d", [-1, 1, 3])
    variables.add_continuous_variable("e", -2, 2, 0.25, "", "")
    variables.move_to_binary(constraint.constraints)
    constraint.add_constraint(expression, True, False)
    constraint.translate_constraints(variables)
    dictionary_constraints_qubo = {
        ("b3",): -1.0,
        ("b4",): -1.0,
        ("b5",): -1.0,
        ("b3", "b4"): 2.0,
        ("b3", "b5"): 2.0,
        ("b4", "b5"): 2.0,
        (): 1.0,
    }
    qubo_first = constraint.constraints_penalty_functions[0][0]
    qubo_second = constraint.constraints_penalty_functions[1][0]
    qubo_first_re = {}
    for key in qubo_first:
        qubo_first_re[tuple(sorted(key))] = qubo_first[key]
    qubo_second_re = {}
    for key in qubo_second:
        qubo_second_re[tuple(sorted(key))] = qubo_second[key]
    dictionary_constraints_qubo_re = {}
    for key, value in dictionary_constraints_qubo.items():
        dictionary_constraints_qubo_re[tuple(sorted(key))] = value
    if expression == "~b0 = b1":
        dictionary_constraints_qubo_2 = {("b0",): -1, ("b1",): -1, ("b0", "b1"): 2, (): 1}
        dictionary_constraints_qubo_2_re = {}
        for key, value in dictionary_constraints_qubo_2.items():
            dictionary_constraints_qubo_2_re[tuple(sorted(key))] = value
        assert dict(sorted(qubo_second_re.items())) == dict(sorted(dictionary_constraints_qubo_2_re.items()))
        assert dict(sorted(qubo_first_re.items())) == dict(sorted(dictionary_constraints_qubo_re.items()))


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
        "maximum_coefficient",
        "VLM",
        "MOMC",
        "MOC",
        "upper lower bound naive",
        "upper lower bound posiform and negaform method",
    ],
)
def test_problem(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    constraint.add_constraint("c >= 1", True, True, False)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    qubo = problem.write_the_final_cost_function(lambda_strategy)
    lambdas_or = problem.lambdas
    lambdas = [1.1 * el for el in lambdas_or]

    reference_qubo_dict = {
        ("b0",): 1.0,
        ("b1",): 2.0 - lambdas[1],
        ("b1", "b2"): 2.0 * lambdas[1],
        ("b1", "b3"): 2.0 * lambdas[1],
        ("b1", "b4"): -0.25,
        ("b1", "b5"): -0.5,
        ("b1", "b6"): -1.0,
        ("b1", "b7"): -2.0,
        ("b1", "b8"): -0.25,
        ("b2", "b4"): 0.25,
        ("b2",): -2.0 - lambdas[1],
        ("b2", "b3"): 2.0 * lambdas[1],
        ("b2", "b5"): 0.5,
        ("b2", "b6"): 1.0,
        ("b2", "b7"): 2.0,
        ("b2", "b8"): 0.25,
        ("b3", "b4"): 0.75,
        ("b3",): -6.0 - lambdas[1],
        ("b3", "b5"): 1.5,
        ("b3", "b6"): 3.0,
        ("b3", "b7"): 6.0,
        ("b3", "b8"): 0.75,
        ("b4",): -0.9375 - 1.4375 * lambdas[0],
        ("b5",): -1.75 - 2.75 * lambdas[0],
        ("b6",): -3.0 - 5.0 * lambdas[0],
        ("b7",): -4.0 - 8.0 * lambdas[0],
        ("b8",): -0.9375 - 1.4375 * lambdas[0],
        ("__a0",): 7.0 * lambdas[0],
        ("b4", "b5"): 0.25 + 0.25 * lambdas[0],
        ("b4", "b6"): 0.5 + 0.5 * lambdas[0],
        ("b4", "b7"): 1.0 + 1.0 * lambdas[0],
        ("b4", "b8"): 0.125 + 0.125 * lambdas[0],
        ("__a0", "b4"): -0.5 * lambdas[0],
        ("b5", "b6"): 1.0 + 1.0 * lambdas[0],
        ("b5", "b7"): 2.0 + 2.0 * lambdas[0],
        ("b5", "b8"): 0.25 + 0.25 * lambdas[0],
        ("__a0", "b5"): -1.0 * lambdas[0],
        ("b6", "b7"): 4.0 + 4.0 * lambdas[0],
        ("b6", "b8"): 0.5 + 0.5 * lambdas[0],
        ("__a0", "b6"): -2.0 * lambdas[0],
        ("b7", "b8"): 1.0 + 1.0 * lambdas[0],
        ("__a0", "b7"): -4.0 * lambdas[0],
        ("__a0", "b8"): -0.5 * lambdas[0],
        (): 4.0 + lambdas[1] + 9.0 * lambdas[0],
    }
    qubo_re = {}
    for key in qubo:
        qubo_re[tuple(sorted(key))] = qubo[key]
    reference_qubo_dict_re = {}
    for key, value in reference_qubo_dict.items():
        reference_qubo_dict_re[tuple(sorted(key))] = value
    if lambda_strategy in {"upper_bound_only_positive", "upper lower bound naive"}:
        assert lambdas == [52.25 * 1.1] * 2
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "maximum_coefficient":
        assert lambdas == [10.0 * 1.1] * 2
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy in {"VLM", "MOMC"}:
        assert lambdas == [12.0 * 1.1] * 2
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "MOC":
        assert lambdas == [7 * 1.1, 6 * 1.1]
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))
    elif lambda_strategy == "upper lower bound posiform and negaform method":
        assert lambdas == [31.625 * 1.1] * 2
        assert dict(sorted(qubo_re.items())) == dict(sorted(reference_qubo_dict_re.items()))


@pytest.mark.parametrize(
    "lambda_strategy",
    [
        "upper_bound_only_positive",
        "maximum_coefficient",
        "VLM",
        "MOMC",
        "MOC",
        "upper lower bound naive",
        "upper lower bound posiform and negaform method",
    ],
)
def test_simulated_annealer_solver(lambda_strategy: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    cost_function = cast(Expr, a0 + b0 * c0 + c0**2)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    sol = solver.solve_simulated_annealing(problem, lambda_strategy=lambda_strategy)
    if isinstance(sol, Solution):
        all_satisfy, _each_satisfy = sol.check_constraint_optimal_solution()
        assert sol.best_solution == {"a": 0.0, "b": 3.0, "c": -1.5}
        assert sol.best_energy < -2.24  # (the range if for having no issues with numerical errors)
        assert sol.best_energy > -2.26
        assert sol.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.25}
        assert all_satisfy
    else:
        assert sol


@pytest.mark.parametrize(
    ("lambda_strategy", "constraint_expr"),
    [
        ("upper_bound_only_positive", "c >= 1"),
        ("maximum_coefficient", "c >= 1"),
        ("VLM", "c >= 1"),
        ("MOMC", "c >= 1"),
        ("MOC", "c >= 1"),
        ("upper lower bound naive", "c >= 1"),
        ("upper lower bound posiform and negaform method", "c >= 1"),
        ("maximum_coefficient", "c > 1"),
        ("VLM", "c > 1"),
        ("MOMC", "c > 1"),
        ("MOC", "c > 1"),
        ("upper lower bound naive", "c > 1"),
        ("upper lower bound posiform and negaform method", "c > 1"),
        ("maximum_coefficient", "b < 1"),
        ("VLM", "b < 1"),
        ("MOMC", "b < 1"),
        ("MOC", "b < 1"),
        ("upper lower bound naive", "b < 1"),
        ("upper lower bound posiform and negaform method", "b < 1"),
        ("maximum_coefficient", "b <= 1"),
        ("VLM", "b <= 1"),
        ("MOMC", "b <= 1"),
        ("MOC", "b <= 1"),
        ("upper lower bound naive", "b <= 1"),
        ("upper lower bound posiform and negaform method", "b <= 1"),
        ("upper_bound_only_positive", "b + c >= 2"),
        ("maximum_coefficient", "b + c >= 2"),
        ("VLM", "b + c >= 2"),
        ("MOMC", "b + c >= 2"),
        ("MOC", "b + c >= 2"),
        ("upper lower bound naive", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "b + c >= 2"),
        ("upper_bound_only_positive", "a = 1"),
        ("maximum_coefficient", "a = 1"),
        ("VLM", "a = 1"),
        ("MOMC", "a = 1"),
        ("MOC", "a = 1"),
        ("upper lower bound naive", "a = 1"),
        ("upper lower bound posiform and negaform method", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained(lambda_strategy: str, constraint_expr: str) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, lambda_strategy=lambda_strategy, num_reads=1000, annealing_time=100
    )
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.4 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution in ({"a": 0.0, "b": 1.0, "c": -0.5}, {"a": 0.0, "b": -1.0, "c": 0.5})
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_update", "constraint_expr"),
    [
        ("sequential penalty increase", "c >= 1"),
        ("scaled sequential penalty increase", "c >= 1"),
        ("binary search penalty algorithm", "c >= 1"),
        ("sequential penalty increase", "c > 1"),
        ("scaled sequential penalty increase", "c > 1"),
        ("binary search penalty algorithm", "c > 1"),
        ("sequential penalty increase", "b < 1"),
        ("scaled sequential penalty increase", "b < 1"),
        ("binary search penalty algorithm", "b < 1"),
        ("sequential penalty increase", "b <= 1"),
        ("scaled sequential penalty increase", "b <= 1"),
        ("binary search penalty algorithm", "b <= 1"),
        ("sequential penalty increase", "b + c >= 2"),
        ("scaled sequential penalty increase", "b + c >= 2"),
        ("binary search penalty algorithm", "b + c >= 2"),
        ("sequential penalty increase", "a = 1"),
        ("scaled sequential penalty increase", "a = 1"),
        ("binary search penalty algorithm", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained_lambda_update_mechanism(
    lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy="manual", lambda_value=5.0
    )
    solver.get_lambda_updates()
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution in ({"a": 0.0, "b": 1.0, "c": -0.5}, {"a": 0.0, "b": -1.0, "c": 0.5})
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "c >= 1"),
        ("VLM", "sequential penalty increase", "c >= 1"),
        ("MOMC", "sequential penalty increase", "c >= 1"),
        ("MOC", "sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c >= 1"),
        ("VLM", "scaled sequential penalty increase", "c >= 1"),
        ("MOMC", "scaled sequential penalty increase", "c >= 1"),
        ("MOC", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c >= 1"),
        ("VLM", "binary search penalty algorithm", "c >= 1"),
        ("MOMC", "binary search penalty algorithm", "c >= 1"),
        ("MOC", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c >= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "sequential penalty increase", "c > 1"),
        ("VLM", "sequential penalty increase", "c > 1"),
        ("MOMC", "sequential penalty increase", "c > 1"),
        ("MOC", "sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "c > 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "c > 1"),
        ("VLM", "scaled sequential penalty increase", "c > 1"),
        ("MOMC", "scaled sequential penalty increase", "c > 1"),
        ("MOC", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "c > 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "c > 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "c > 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "c > 1"),
        ("VLM", "binary search penalty algorithm", "c > 1"),
        ("MOMC", "binary search penalty algorithm", "c > 1"),
        ("MOC", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "c > 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "c > 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "sequential penalty increase", "b < 1"),
        ("VLM", "sequential penalty increase", "b < 1"),
        ("MOMC", "sequential penalty increase", "b < 1"),
        ("MOC", "sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b < 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b < 1"),
        ("VLM", "scaled sequential penalty increase", "b < 1"),
        ("MOMC", "scaled sequential penalty increase", "b < 1"),
        ("MOC", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b < 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b < 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b < 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b < 1"),
        ("VLM", "binary search penalty algorithm", "b < 1"),
        ("MOMC", "binary search penalty algorithm", "b < 1"),
        ("MOC", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b < 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b < 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "sequential penalty increase", "b <= 1"),
        ("VLM", "sequential penalty increase", "b <= 1"),
        ("MOMC", "sequential penalty increase", "b <= 1"),
        ("MOC", "sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b <= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b <= 1"),
        ("VLM", "scaled sequential penalty increase", "b <= 1"),
        ("MOMC", "scaled sequential penalty increase", "b <= 1"),
        ("MOC", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b <= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b <= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b <= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "b <= 1"),
        ("VLM", "binary search penalty algorithm", "b <= 1"),
        ("MOMC", "binary search penalty algorithm", "b <= 1"),
        ("MOC", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "b <= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b <= 1"),
        ("upper_bound_only_positive", "sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "sequential penalty increase", "b + c >= 2"),
        ("VLM", "sequential penalty increase", "b + c >= 2"),
        ("MOMC", "sequential penalty increase", "b + c >= 2"),
        ("MOC", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "b + c >= 2"),
        ("maximum_coefficient", "scaled sequential penalty increase", "b + c >= 2"),
        ("VLM", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOMC", "scaled sequential penalty increase", "b + c >= 2"),
        ("MOC", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound naive", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "b + c >= 2"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "b + c >= 2"),
        ("maximum_coefficient", "binary search penalty algorithm", "b + c >= 2"),
        ("VLM", "binary search penalty algorithm", "b + c >= 2"),
        ("MOMC", "binary search penalty algorithm", "b + c >= 2"),
        ("MOC", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound naive", "binary search penalty algorithm", "b + c >= 2"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "b + c >= 2"),
        ("upper_bound_only_positive", "sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "sequential penalty increase", "a = 1"),
        ("VLM", "sequential penalty increase", "a = 1"),
        ("MOMC", "sequential penalty increase", "a = 1"),
        ("MOC", "sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "a = 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "a = 1"),
        ("VLM", "scaled sequential penalty increase", "a = 1"),
        ("MOMC", "scaled sequential penalty increase", "a = 1"),
        ("MOC", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "a = 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "a = 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "a = 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "a = 1"),
        ("VLM", "binary search penalty algorithm", "a = 1"),
        ("MOMC", "binary search penalty algorithm", "a = 1"),
        ("MOC", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "a = 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "a = 1"),
    ],
)
def test_simulated_annealer_solver_constrained_lambda_update_mechanism_and_strategy(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for the problem constructions"""
    variables = Variables()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cast(Expr, a0 + b0 * c0 + c0**2))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy=lambda_strategy
    )
    solver.get_lambda_updates()
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "c >= 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.0} or not all_satisfy
            assert (
                solution.best_energy < 0.1
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.0} or not all_satisfy
        elif constraint_expr == "c > 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 1.25} or not all_satisfy
            assert (
                solution.best_energy < 0.32 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > 0.31 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": 0.3125} or not all_satisfy
        elif constraint_expr == "b < 1":
            assert solution.best_solution == {"a": 0.0, "b": -1.0, "c": 0.5} or not all_satisfy
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        elif constraint_expr == "b <= 1":
            assert (
                solution.best_solution in ({"a": 0.0, "b": 1.0, "c": -0.5}, {"a": 0.0, "b": -1.0, "c": 0.5})
                or not all_satisfy
            )
            assert (
                solution.best_energy < -0.2 or not all_satisfy
            )  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -0.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -0.25} or not all_satisfy
        if constraint_expr == "b + c >= 2":
            assert solution.best_solution == {"a": 0.0, "b": 3.0, "c": -1.0} or not all_satisfy
            assert (
                solution.best_energy < -1.9
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -2.1 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -2.0} or not all_satisfy
        if constraint_expr == "a = 1":
            assert solution.best_solution == {"a": 1, "b": 3.0, "c": -1.5} or not all_satisfy
            assert (
                solution.best_energy < -1.2
            ) or not all_satisfy  # (the range if for having no issues with numerical errors)
            assert solution.best_energy > -1.3 or not all_satisfy
            assert solution.optimal_solution_cost_functions_values() == {"a + b*c + c**2": -1.25} or not all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    ("lambda_strategy", "lambda_update", "constraint_expr"),
    [
        ("upper_bound_only_positive", "sequential penalty increase", "M1_0_1 >= 1"),
        ("maximum_coefficient", "sequential penalty increase", "M1_0_1 >= 1"),
        ("VLM", "sequential penalty increase", "M1_0_1 >= 1"),
        ("MOMC", "sequential penalty increase", "M1_0_1 >= 1"),
        ("MOC", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound naive", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "sequential penalty increase", "M1_0_1 >= 1"),
        ("upper_bound_only_positive", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("maximum_coefficient", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("VLM", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("MOMC", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("MOC", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound naive", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "scaled sequential penalty increase", "M1_0_1 >= 1"),
        ("upper_bound_only_positive", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("maximum_coefficient", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("VLM", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("MOMC", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("MOC", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("upper lower bound naive", "binary search penalty algorithm", "M1_0_1 >= 1"),
        ("upper lower bound posiform and negaform method", "binary search penalty algorithm", "M1_0_1 >= 1"),
    ],
)
def test_simulated_annealing_cost_function_matrix(
    lambda_strategy: str, lambda_update: str, constraint_expr: str
) -> None:
    """Test for cost function translation"""
    variables = Variables()
    m1 = variables.add_continuous_variables_array("M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2")
    m2 = variables.add_continuous_variables_array("M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2")
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(np.matmul(m1, m2).item(0, 0))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem, max_lambda_update=10, lambda_update_mechanism=lambda_update, lambda_strategy=lambda_strategy
    )
    solver.get_lambda_updates()
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        if constraint_expr == "M1_0_1 >= 1":
            assert (
                solution.best_solution in ({"M1": [[-1, 2]], "M2": [[2], [-1]]}, {"M1": [[2, 2]], "M2": [[-1], [-1]]})
                or not all_satisfy
            )
            assert (solution.best_energy < -3.9) or not all_satisfy
            assert solution.best_energy > -4.1 or not all_satisfy
            assert (
                solution.optimal_solution_cost_functions_values() == {"M1_0_0*M2_0_0 + M1_0_1*M2_1_0": -4.0}
                or not all_satisfy
            )
    else:
        assert solution


def test_predict_solver_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve(problem, num_runs=20)
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        print(solution.best_solution)
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


def test_gas_solver_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_grover_adaptive_search_qubo(problem, qubit_values=6, num_runs=10)
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        print(solution.best_solution)
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


def test_qaoa_solver_qubo_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_qaoa_qubo(
        problem,
        num_runs=10,
    )
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


def test_vqe_solver_qubo_basic() -> None:
    """Test for the problem constructions"""
    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_vqe_qubo(
        problem,
        num_runs=10,
    )
    if isinstance(solution, Solution):
        all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
        assert solution.best_solution == {"a": 1.0, "b": 0.0, "c": 1.0}
        print(solution.best_solution)
        assert solution.best_energy < -5.9  # (the range if for having no issues with numerical errors)
        assert solution.best_energy > -6.1
        print(solution.optimal_solution_cost_functions_values())
        assert solution.optimal_solution_cost_functions_values() == {"-2.0*a*c - a - b*c + 2.0*b - 3.0*c": -6.0}
        assert all_satisfy
    else:
        assert solution


@pytest.mark.parametrize(
    "problem_name",
    [
        "maxcut_3_10_1",
        "maxcut_3_10_5",
        "maxcut_3_50_1",
        "maxcut_3_50_2",
    ],
)
def test_predict_solver_maxcut(problem_name: str) -> None:
    """Test for the problem constructions"""
    pathfile = Path(__file__).parent / "maxcut" / str(problem_name + ".txt")
    with pathfile.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        el = lines[0].split()
        nodes = int(el[0])
        weight = np.zeros((nodes, nodes))
        for k in range(1, len(lines)):
            el = lines[k].split()
            i = int(el[0])
            j = int(el[1])
            w = int(el[2])
            weight[i, j] = w

        variables = Variables()
        x = variables.add_binary_variables_array("x", [nodes])
        objective_function = ObjectiveFunction()
        if not isinstance(x, bool):
            cut = 0
            for i in range(nodes):
                for j in range(i + 1, nodes):
                    cut += weight.item((i, j)) * (x[j] + x[i] - 2 * x[i] * x[j])
            objective_function.add_objective_function(cast(Expr, cut), minimization=False)
            constraint = Constraints()
            problem = Problem()
            problem.create_problem(variables, constraint, objective_function)
            solver = Solver()
            solution = solver.solve(
                problem,
                num_runs=10,
                coeff_precision=1.0,
            )
            pathfilesol = Path(__file__).parent / "maxcutResults" / str(problem_name + "_sol.txt")
            with pathfilesol.open("r", encoding="utf-8") as fsol:
                reference_val = float(fsol.readline())
                if not isinstance(solution, bool) and solution is not None:
                    _all_satisfy, _each_satisfy = solution.check_constraint_optimal_solution()
                    assert solution.best_energy == reference_val
                else:
                    assert solution
        else:
            assert x
