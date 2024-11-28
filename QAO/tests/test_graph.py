"""File for making some tests during the Development"""

from __future__ import annotations

from typing import Any

import networkx as nx

# for managing symbols
from mqt.qao.karp import KarpGraphs
from mqt.qao.problem import Problem
from io import StringIO
import sys


def test_clique_initialization():
    """Test the initialization of the clique problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k: int = 3
    problem = KarpGraphs.clique(graph, k=k, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for clique initialization"

def test_print_solution():
    """Unit test for the print_solution method."""
    
    # Capture printed output
    captured_output = StringIO()
    sys.stdout = captured_output

    # Test data
    problem_name = "Test Problem"
    file_name = "test_file"
    solution = "This is the solution."
    summary = "Summary details."

    # Expected output
    expected_output = (
        "Test Problemtest_file\n"
        "=====================\n"
        "This is the solution.\n"
        "---------------------\n"
        "Summary details.\n"
    )

    # Call the method
    KarpGraphs.print_solution(
        problem_name=problem_name,
        file_name=file_name,
        solution=solution,
        summary=summary
    )

    # Reset stdout and check the output
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output == expected_output, "The printed output does not match the expected result."




def test_clique_solving_basic():
    """Test the basic solving of the clique problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k: int = 3
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), (
        "Each element in the solution should be an integer representing a node"
    )


def test_clique_solution_k_value():
    """Test that the solution has the correct number of nodes for the clique size k."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (2, 4), (4, 3)])
    k: int = 3
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) == k, f"Expected a clique of size {k}, but got {len(solution)}"


def test_clique_solution_maximal_clique():
    """Test finding the maximal clique (k=0) in the graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 1)])
    solution = KarpGraphs.clique(graph, k=0, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) == 3, "Expected maximal clique of size 3 for this input graph"


def test_clique_solution_validation_correct():
    """Test validation for a correct clique solution."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (2, 4)])
    solution: list[Any] = [1, 2, 3]
    validation = KarpGraphs.check_clique_solution(graph, solution)

    assert validation["Valid Solution"], "Expected solution to be valid"


def test_clique_solution_validation_incorrect():
    """Test validation for an incorrect clique solution (not a complete subgraph)."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (1, 4)])
    solution: list[Any] = [1, 2, 4]
    validation = KarpGraphs.check_clique_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing edge"


def test_clique_empty_graph():
    """Test handling of an empty graph."""
    graph = nx.Graph()
    k: int = 3
    problem = KarpGraphs.clique(graph, k=k, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance even with an empty graph"


def test_clique_single_node_graph():
    """Test a graph with a single node."""
    graph = nx.Graph()
    graph.add_node(1)
    k: int = 1
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [1], "Expected the single node as the solution"


def test_clique_large_k_value():
    """Test the clique method with a k value larger than the graph's maximum clique size."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    k: int = 2
    solution = KarpGraphs.clique(graph, k=k, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) == k, "Expected an empty solution as no clique of size 5 exists in the graph"


def test_clique_cover_initialization():
    """Test the initialization of the clique cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors: int = 2
    problem = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=False)

    assert isinstance(problem, Problem), "Expected a Problem instance for clique cover initialization"


def test_clique_cover_solving_basic():
    """Test the basic solving of the clique cover problem."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors: int = 2
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(len(pair) == 2 for pair in solution), (
        "Each element in the solution should be a tuple representing (node, color)"
    )


def test_clique_cover_solution_num_colors():
    """Test that the solution uses the correct number of colors."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)])
    num_colors: int = 2
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    used_colors = {color for _, color in solution}
    assert len(used_colors) <= num_colors, (
        f"Expected solution to use up to {num_colors} colors, but used {len(used_colors)}"
    )


def test_clique_cover_solution_correct_cover():
    """Test validation for a correct clique cover solution."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    solution: list[Any] = [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2)]  # known correctcover
    validation = KarpGraphs.check_clique_cover_solution(graph, solution)

    assert validation["Valid Solution"], "Expected solution to be valid for correct clique cover"


def test_clique_cover_solution_incorrect_cover():
    """Test validation for an incorrect clique cover solution (non-clique assignment)."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (1, 4)])
    solution: list[Any] = [(1, 1), (2, 1), (3, 2), (4, 1)]  # Invalid cover (node 4 doesn't connect to 2 or 3)
    validation = KarpGraphs.check_clique_cover_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected solution to be invalid due to non-clique assignment"


def test_clique_cover_single_node_graph():
    """Test a graph with a single node."""
    graph = nx.Graph()
    graph.add_node(1)
    num_colors = 1
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [(1, 1)], "Expected the single node with color 1 in the solution"


def test_clique_cover_large_num_colors():
    """Test the clique cover with a large number of colors, exceeding the minimum requirement."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5)])
    num_colors = 5  # more colors than required
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len({color for _, color in solution}) <= num_colors, (
        "Expected the number of colors used to be within the specified num_colors"
    )


def test_clique_cover_disconnected_graph():
    """Test the clique cover with a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    num_colors = 3
    solution = KarpGraphs.clique_cover(graph, num_colors=num_colors, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_vertex_cover_solution_minimum_size():
    """Test that the solution provides a minimum vertex cover size."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [], "Expected no nodes in the cover as a single node with no edges does not require covering"


def test_vertex_cover_large_num_nodes():
    """Test the vertex cover on a graph with many nodes but few edges."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) <= 3, "Expected the vertex cover to include at most one node per edge"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_vertex_cover_disconnected_graph():
    """Test the vertex cover with a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6)])
    solution = KarpGraphs.vertex_cover(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(len(node_color) == 2 for node_color in solution), (
        "Each element in the solution should be a tuple of (node, color)"
    )


def test_graph_coloring_validity():
    """Test that the coloring solution is valid for a simple graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=3, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid coloring"


def test_graph_coloring_invalid_colors():
    """Test that the graph coloring fails when not enough colors are provided."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert not validation["Valid Solution"], "Expected the solution to be invalid with insufficient colors"


def test_graph_coloring_chromatic_number():
    """Test that the solution respects the chromatic number of a small cycle graph."""
    graph = nx.cycle_graph(4)
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid with 2 colors for C4 graph"


def test_graph_coloring_disconnected_graph():
    """Test coloring on a disconnected graph with 2 components."""
    graph = nx.Graph([(1, 2), (3, 4)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=2, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid with disconnected components"


def test_graph_coloring_single_node():
    """Test coloring on a single-node graph."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.graph_coloring(graph, num_colors=1, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [(1, 1)], "Expected the single node to be colored with the only available color"


def test_graph_coloring_complete_graph():
    """Test coloring on a complete graph with n nodes, requiring n colors."""
    n = 4
    graph = nx.complete_graph(n)
    solution = KarpGraphs.graph_coloring(graph, num_colors=n, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_graph_coloring_solution(graph, solution)
    assert validation["Valid Solution"], f"Expected a valid coloring with {n} colors for K{n} graph"


def test_graph_coloring_large_sparse_graph():
    """Test graph coloring on a large sparse graph with fewer colors than nodes."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
    solution = KarpGraphs.graph_coloring(graph, num_colors=3, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_independent_set_validity():
    """Test that the independent set solution is valid for a simple graph."""
    graph = nx.Graph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid independent set"


def test_independent_set_disconnected_graph():
    """Test independent set on a disconnected graph."""
    graph = nx.Graph([(1, 2), (3, 4)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for a disconnected graph"


def test_independent_set_large_sparse_graph():
    """Test independent set on a large sparse graph."""
    graph = nx.Graph([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)])
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for a sparse graph"


def test_independent_set_single_node():
    """Test independent set on a single-node graph."""
    graph = nx.Graph()
    graph.add_node(1)
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [1], "Expected the single node to be part of the independent set"


def test_independent_set_complete_graph():
    """Test independent set on a complete graph, where only one node can be in the set."""
    n = 4
    graph = nx.complete_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) == 1, "Expected only one node in the independent set for a complete graph"


def test_independent_set_path_graph():
    """Test independent set on a path graph, where alternate nodes form the maximum independent set."""
    n = 5
    graph = nx.path_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_independent_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected a valid solution for the path graph"
    assert len(solution) == (n + 1) // 2, "Expected maximum independent set to have (n+1)//2 nodes for path graph"


def test_independent_set_cycle_graph():
    """Test independent set on a cycle graph, where alternate nodes form the maximum independent set."""
    n = 6
    graph = nx.cycle_graph(n)
    solution = KarpGraphs.independent_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(node, int) for node in solution), "Each element in the solution should be an integer node"


def test_directed_feedback_vertex_set_validity():
    """Test that the feedback vertex set solution is valid for a simple graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid feedback vertex set"


def test_directed_feedback_vertex_set_minimal_size():
    """Test that the solution provides a minimal feedback vertex set."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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


def test_directed_feedback_vertex_set_multiple_cycles():
    """Test the feedback vertex set on a graph with multiple cycles."""
    graph = nx.DiGraph([(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_directed_feedback_vertex_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be valid for multiple cycles"

    assert len(solution) >= 1, "Expected the feedback vertex set to include at least one node"


def test_directed_feedback_vertex_set_disconnected_graph():
    """Test the feedback vertex set on a disconnected graph."""
    graph = nx.DiGraph([(1, 2), (2, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_vertex_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

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
    assert all(len(edge) == 2 for edge in solution), (
        "Each element in the solution should be a tuple representing an edge"
    )


def test_directed_feedback_edge_set_validity():
    """Test that the feedback edge set solution is valid for a simple graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)
    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)
    assert validation["Valid Solution"], "Expected the solution to be a valid feedback edge set"


def test_directed_feedback_edge_set_minimal_size():
    """Test that the solution provides a minimal feedback edge set."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert len(solution) == 1, "Expected the minimal feedback edge set to have size 1"


def test_directed_feedback_edge_set_solution_correct():
    """Test validation for a correct feedback edge set solution."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [(3, 1)]

    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)

    assert validation["Valid Solution"], "Expected a valid solution for the disconnected graph"


def test_directed_feedback_edge_set_solution_incorrect():
    """Test validation for an incorrect feedback edge set solution (cycles remain)."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    solution = [(1, 3)]
    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)

    assert not validation["Valid Solution"], "Expected a valid solution for the disconnected graph"


def test_directed_feedback_edge_set_no_cycles():
    """Test the feedback edge set on an acyclic graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [], "Expected an empty feedback edge set for an acyclic graph"


def test_directed_feedback_edge_set_multiple_cycles():
    """Test the feedback edge set on a graph with multiple cycles."""
    graph = nx.DiGraph([(1, 2), (2, 1), (2, 3), (3, 2)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)
    validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)

    assert validation["Valid Solution"], "Expected the solution to be valid for multiple cycles"


def test_directed_feedback_edge_set_disconnected_graph():
    """Test the feedback edge set on a disconnected graph."""
    graph = nx.DiGraph([(1, 2), (2, 1), (3, 4)])
    solution = KarpGraphs.directed_feedback_edge_set(graph, solve=True)

    if isinstance(solution, list):
        validation = KarpGraphs.check_directed_feedback_edge_set_solution(graph, solution)
        assert validation["Valid Solution"], "Expected a valid solution for the disconnected graph"
    else:
        msg = "Expected list[tuple[int, int]] but got another type"
        raise TypeError(msg)

    assert validation["Valid Solution"], "Expected the solution to be valid for disconnected graph"


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
