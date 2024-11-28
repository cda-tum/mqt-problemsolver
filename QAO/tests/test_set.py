from __future__ import annotations

import sys
from io import StringIO
from typing import Any

import networkx as nx

# for managing symbols
from mqt.qao.karp import KarpSets
from mqt.qao.problem import Problem


def test_set_cover_initialization():
    """Verify that initializing a set cover problem returns a Problem instance without solving it."""
    input_data = [(1, [1, 2]), (2, [2, 3])]
    problem = KarpSets.set_cover(input_data, solve=False)
    assert isinstance(problem, Problem)


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
        "Test Problemtest_file\n=====================\nThis is the solution.\n---------------------\nSummary details.\n"
    )

    # Call the method
    KarpSets.print_solution(problem_name=problem_name, file_name=file_name, solution=solution, summary=summary)

    # Reset stdout and check the output
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output == expected_output, "The printed output does not match the expected result."


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
    input_data: list[Any] = []
    problem = KarpSets.set_cover(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_set_packing_initialization():
    """Verify that initializing a set packing problem returns a Problem instance without solving it."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3])]
    problem = KarpSets.set_packing(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance"


def test_set_packing_solving_basic():
    """Ensure that solving a basic set packing problem returns a list of tuples as the solution."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3])]
    solution = KarpSets.set_packing(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) for item in solution), "Each solution item should be a tuple"


def test_set_packing_solution_validation_correct():
    """Check that a known valid solution for set packing is validated correctly."""
    input_data: list[Any] = [(1, [1, 2]), (2, [3, 4]), (1, [4])]
    solution = [(1, [1, 2]), (1, [4])]
    validation = KarpSets.check_set_packing_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_set_packing_solution_validation_extraneous():
    """Confirm that a solution with extraneous sets for set packing is marked invalid."""
    input_data: list[Any] = [(1, [1, 2]), (2, [3, 4])]
    solution = [(1, [1, 2]), (2, [3, 4]), (3, [1, 3])]
    validation = KarpSets.check_set_packing_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous sets"


def test_set_packing_empty_input():
    """Ensure that initializing a set packing problem with empty input returns a Problem instance."""
    input_data: list[Any] = []
    problem = KarpSets.set_packing(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_exact_cover_initialization():
    """Verify that initializing an exact cover problem returns a Problem instance without solving it."""
    input_data: list[Any] = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    problem = KarpSets.exact_cover(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance"


def test_exact_cover_solving_basic():
    """Ensure that solving a basic exact cover problem returns a list of tuples as the solution."""
    input_data: list[Any] = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = KarpSets.exact_cover(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, tuple) for item in solution), "Each solution item should be a tuple"


def test_exact_cover_solution_validation_correct():
    """Check that a known valid solution for exact cover is validated correctly."""
    input_data: list[Any] = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2]), (1, [3, 4])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_exact_cover_solution_validation_incomplete():
    """Verify that an incomplete exact cover solution is marked invalid due to missing elements."""
    input_data: list[Any] = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing elements"


def test_exact_cover_solution_validation_extraneous():
    """Confirm that a solution with extraneous sets for exact cover is marked invalid."""
    input_data: list[Any] = [(1, [1, 2]), (1, [2, 3]), (1, [3, 4])]
    solution = [(1, [1, 2]), (1, [3, 4]), (2, [1, 3])]
    validation = KarpSets.check_exact_cover_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous sets"


def test_exact_cover_empty_input():
    """Ensure that initializing an exact cover problem with empty input returns a Problem instance."""
    input_data: list[Any] = []
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
    solution: list[Any] = [(1, 3, 5)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_three_d_matching_solution_validation_incomplete():
    """Verify that an incomplete 3D matching solution is identified as invalid due to repeated elements."""
    solution = [(1, 2, 3)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to repeated elements"


def test_three_d_matching_solution_validation_extraneous():
    """Confirm that a solution with extraneous triples is identified as invalid."""
    solution: list[Any] = [(1, 2, 3), (3, 2, 1)]
    validation = KarpSets.check_three_d_matching(x, y, z, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous triples"


def test_three_d_matching_solving_with_graph():
    """Solving a 3D matching problem with a graph returns a list of valid triples."""
    solution = KarpSets.three_d_matching(graph, x, y, z, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert all(isinstance(item, list) and len(item) == 3 for item in solution), (
        "Each solution item should be a tuple of length 3"
    )


def test_hitting_set_initialization():
    """Test the initialization of the hitting set problem."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    problem = KarpSets.hitting_set(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for hitting set initialization"


def test_hitting_set_solving_basic():
    """Test the basic solving of the hitting set problem without weights."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution = KarpSets.hitting_set(input_data, solve=True)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(item, int) for item in solution), "Each solution item should be an integer (element index)"


def test_hitting_set_solution_validation_correct():
    """Test the validation of a correct hitting set solution."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution: list[Any] = [1, 3]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_hitting_set_solution_validation_incomplete():
    """Test the validation of an incomplete hitting set solution."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    solution: list[Any] = [1]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to missing elements"


def test_hitting_set_solution_validation_extraneous():
    """Test the validation of a hitting set solution with extraneous elements."""
    input_data: list[Any] = [(1, [1, 2]), (2, [2, 3])]
    solution: list[Any] = [1, 2, 4]
    validation = KarpSets.check_hitting_set_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to extraneous elements"


def test_hitting_set_empty_input():
    """Test the handling of an empty input for the hitting set problem."""
    input_data: list[Any] = []
    problem = KarpSets.hitting_set(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"
