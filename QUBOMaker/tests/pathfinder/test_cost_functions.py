"""Tests for the correctness of all cost functions of the pathfinder module."""

from __future__ import annotations

import pytest
import sympy as sp

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf

from .utils_test import get_test_graph, paths_to_assignment

TEST_GRAPH = get_test_graph()


def evaluate(
    cost_function: pf.CostFunction, path: dict[sp.Expr, int], encoding: pf.EncodingType, loop: bool, n_paths: int = 1
) -> int:
    """Computes the cost of a given path(s) assignment for a given cost function.

    Args:
        cost_function (pf.CostFunction): The cost function to evaluate.
        path (dict[sp.Expr, int]): The assignment of the path(s).
        encoding (pf.EncodingType): The encoding type of the assignment.
        loop (bool): Indicates if the path is a loop.
        n_paths (int, optional): The number of paths. Defaults to 1.

    Returns:
        int: The cost of the assignment for the given cost function.
    """
    settings = pf.PathFindingQUBOGeneratorSettings(encoding, n_paths, TEST_GRAPH.n_vertices, loop)
    formula = cost_function.get_formula(TEST_GRAPH, settings)
    assignment: list[tuple[sp.Expr, sp.Expr | int | float]] = [
        (cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, TEST_GRAPH.n_vertices + 1, i + 1), 0)
        for p in range(settings.n_paths)
        for i in range(settings.max_path_length + 1)
    ]  # x_{p, |V| + 1, i} = 0 for all p, i
    assignment += [
        (
            cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, settings.max_path_length + 1),
            cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, 1)
            if settings.loops
            else sp.Integer(0)
            if settings.encoding_type != pf.EncodingType.BINARY
            else cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, settings.max_path_length),
        )
        for p in range(settings.n_paths)
        for v in range(TEST_GRAPH.n_vertices)
    ]  # x_{p, v, N + 1} = x_{p, v, 1} for all p, v if loop, otherwise 0
    assignment += [
        (cf.FormulaHelpers.adjacency(i + 1, j + 1), TEST_GRAPH.adjacency_matrix[i, j])
        for i in range(TEST_GRAPH.n_vertices)
        for j in range(TEST_GRAPH.n_vertices)
    ]
    return int(
        formula.subs(path)
        .doit()
        .subs(path)
        .doit()
        .subs(dict(assignment))
        .doit()
        .subs(path)
        .subs(dict(assignment))
        .subs(path),
    )


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_position_is(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathPositionIs cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)

    assert evaluate(pf.PathPositionIs(2, [3], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathPositionIs(2, [2, 3], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathPositionIs(2, [1, 5], 2), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathPositionIs(1, [4], 1), encoding_a, encoding_type, loop) > 0
    assert evaluate(pf.PathPositionIs(1, [4, 3, 2], 1), encoding_a, encoding_type, loop) > 0
    assert evaluate(pf.PathPositionIs(1, [1], 2), encoding_b, encoding_type, loop, n_paths=2) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_starts_at_ends_at(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathStartsAt and PathEndsAt cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)

    assert evaluate(pf.PathStartsAt([1, 4], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathStartsAt([1], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathStartsAt([1], 1), encoding_b, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathStartsAt([2], 2), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathEndsAt([3, 5], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathEndsAt([5], 1), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathEndsAt([4], 1), encoding_b, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathEndsAt([5], 2), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathStartsAt([4, 3, 5], 1), encoding_a, encoding_type, loop) > 0
    assert evaluate(pf.PathStartsAt([2], 1), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PathStartsAt([5], 2), encoding_b, encoding_type, loop, n_paths=2) > 0

    assert evaluate(pf.PathEndsAt([1, 2, 3, 4], 1), encoding_a, encoding_type, loop) > 0
    assert evaluate(pf.PathEndsAt([5], 1), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PathEndsAt([4], 2), encoding_b, encoding_type, loop, n_paths=2) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_vertices_exactly_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsVerticesExactlyOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsVerticesExactlyOnce([1, 4], [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathContainsVerticesExactlyOnce([3], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathContainsVerticesExactlyOnce([4], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PathContainsVerticesExactlyOnce([5], [1]), encoding_a, encoding_type, loop) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_vertices_at_least_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsVerticesAtLeastOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsVerticesAtLeastOnce([1, 4], [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathContainsVerticesAtLeastOnce([3], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathContainsVerticesAtLeastOnce([5], [1]), encoding_a, encoding_type, loop) == 0

    assert evaluate(pf.PathContainsVerticesAtLeastOnce([4], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_vertices_at_most_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsVerticesAtMostOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsVerticesAtMostOnce([1, 4], [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathContainsVerticesAtMostOnce([3], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathContainsVerticesAtMostOnce([4], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathContainsVerticesAtMostOnce([5], [1]), encoding_a, encoding_type, loop) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_edges_exactly_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsEdgesExactlyOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsEdgesExactlyOnce([(5, 4), (4, 1)], [1]), encoding_a, encoding_type, loop) == 0
    assert (
        evaluate(pf.PathContainsEdgesExactlyOnce([(1, 3), (3, 4)], [1]), encoding_b, encoding_type, loop, n_paths=2)
        == 0
    )

    assert evaluate(pf.PathContainsEdgesExactlyOnce([(1, 3)], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PathContainsEdgesExactlyOnce([(1, 5)], [1]), encoding_a, encoding_type, loop) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_edges_at_least_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsEdgesAtLeastOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsEdgesAtLeastOnce([(5, 4), (4, 1)], [1]), encoding_a, encoding_type, loop) == 0
    assert (
        evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 3), (3, 4)], [1]), encoding_b, encoding_type, loop, n_paths=2)
        == 0
    )
    assert evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 5)], [1]), encoding_a, encoding_type, loop) == 0

    assert evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 3)], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_path_contains_edges_at_most_once(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathContainsEdgesAtMostOnce cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathContainsEdgesAtMostOnce([(5, 4), (4, 1), (5, 5)], [1]), encoding_a, encoding_type, loop) == 0
    assert (
        evaluate(pf.PathContainsEdgesAtMostOnce([(1, 3), (3, 4)], [1]), encoding_b, encoding_type, loop, n_paths=2) == 0
    )
    assert evaluate(pf.PathContainsEdgesAtMostOnce([(1, 3)], [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathContainsEdgesAtMostOnce([(1, 5)], [1]), encoding_a, encoding_type, loop) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_precedence_constraint(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PrecedenceConstraint cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)

    assert evaluate(pf.PrecedenceConstraint(1, 3, [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PrecedenceConstraint(3, 4, [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PrecedenceConstraint(3, 2, [1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PrecedenceConstraint(4, 2, [1]), encoding_b, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PrecedenceConstraint(1, 4, [1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PrecedenceConstraint(2, 4, [1]), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PrecedenceConstraint(5, 2, [2]), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PrecedenceConstraint(4, 3, [1]), encoding_a, encoding_type, loop) > 0
    assert evaluate(pf.PrecedenceConstraint(5, 1, [1]), encoding_a, encoding_type, loop) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_share_no_vertices(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathsShareNoVertices cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 5], [1, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathsShareNoVertices(1, 2), encoding_a, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathsShareNoVertices(1, 2), encoding_b, encoding_type, loop, n_paths=3) == 0
    assert evaluate(pf.PathsShareNoVertices(2, 3), encoding_b, encoding_type, loop, n_paths=3) == 0

    assert evaluate(pf.PathsShareNoVertices(1, 3), encoding_b, encoding_type, loop, n_paths=3) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_share_no_edges(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathsShareNoEdges cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment(
        [[1, 3, 4], [2, 3], [1, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
    )

    assert evaluate(pf.PathsShareNoEdges(1, 2), encoding_a, encoding_type, loop, n_paths=2) == 0
    assert evaluate(pf.PathsShareNoEdges(1, 2), encoding_b, encoding_type, loop, n_paths=3) == 0
    assert evaluate(pf.PathsShareNoEdges(2, 3), encoding_b, encoding_type, loop, n_paths=3) == 0

    assert evaluate(pf.PathsShareNoEdges(1, 3), encoding_b, encoding_type, loop, n_paths=3) > 0


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_maximize_minimize(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the MinimizePathLength and MaximizePathLength cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)

    assert evaluate(pf.MinimizePathLength([1]), encoding_a, encoding_type, loop) == 18 if not loop else 20
    assert evaluate(pf.MinimizePathLength([1]), encoding_b, encoding_type, loop, n_paths=2) == 10 if not loop else 12
    assert evaluate(pf.MinimizePathLength([2]), encoding_b, encoding_type, loop, n_paths=2) == 5 if not loop else 7
    assert evaluate(pf.MinimizePathLength([1, 2]), encoding_b, encoding_type, loop, n_paths=2) == 15 if not loop else 19

    assert evaluate(pf.MaximizePathLength([1]), encoding_a, encoding_type, loop) == -18 if not loop else -20
    assert evaluate(pf.MaximizePathLength([1]), encoding_b, encoding_type, loop, n_paths=2) == -10 if not loop else -12
    assert evaluate(pf.MaximizePathLength([2]), encoding_b, encoding_type, loop, n_paths=2) == -5 if not loop else -7
    assert (
        evaluate(pf.MaximizePathLength([1, 2]), encoding_b, encoding_type, loop, n_paths=2) == -15 if not loop else -19
    )


@pytest.mark.parametrize(
    ("encoding_type", "loop"),
    [
        (cf.EncodingType.ONE_HOT, False),
        (cf.EncodingType.DOMAIN_WALL, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.DOMAIN_WALL, True),
        (cf.EncodingType.BINARY, True),
    ],
)
def test_is_valid(encoding_type: pf.EncodingType, loop: bool) -> None:
    """Test for the correctness of the PathIsValid cost function.

    Args:
        encoding_type (pf.EncodingType): The encoding type to be used.
        loop (bool): Indicates if the cost function should be tested with looping paths.
    """
    encoding_a = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
    encoding_b = paths_to_assignment([[1, 3, 4], [4, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)

    assert evaluate(pf.PathIsValid([1]), encoding_a, encoding_type, loop) == 0
    assert evaluate(pf.PathIsValid([1]), encoding_b, encoding_type, loop, n_paths=2) == 0

    assert evaluate(pf.PathIsValid([2]), encoding_b, encoding_type, loop, n_paths=2) > 0
    assert evaluate(pf.PathIsValid([1, 2]), encoding_b, encoding_type, loop, n_paths=2) > 0


def test_latex_output() -> None:
    """Tests for the correctness of the LaTeX output of custom sympy expressions."""
    printer = sp.StrPrinter()

    s = cf.FormulaHelpers.sum_set(sp.Symbol("x"), ["x"], r"\in V", lambda: [1, 2, 3])
    assert s._latex(printer) == r"\sum_{x \in V} x"  # noqa: SLF001

    a = cf.A(1, 2)
    assert a._latex(printer) == r"A_{1,2}"  # noqa: SLF001

    x = cf.X(1, 2, 3)
    assert x._latex(printer) == r"x_{1,2,3}"  # noqa: SLF001

    d = cf.Decompose(5, 1)
    assert d._latex(printer) == r"\bar{5}_{1}"  # noqa: SLF001


def test_composite_get_formula() -> None:
    """Test for the correctness of the CompositeCostFunction get_formula method."""
    settings = pf.PathFindingQUBOGeneratorSettings(cf.EncodingType.ONE_HOT, 1, TEST_GRAPH.n_vertices, False)
    p1 = cf.PathPositionIs(1, [1], 1)
    p2 = cf.PathPositionIs(2, [2], 1)
    c = cf.CompositeCostFunction((p1, 1), (p2, 1))
    composite_expr = c.get_formula(TEST_GRAPH, settings)
    p1_expr = p1.get_formula(TEST_GRAPH, settings)
    p2_expr = p2.get_formula(TEST_GRAPH, settings)
    assert composite_expr == p1_expr + p2_expr


def test_expanding_sum() -> None:
    """Test for the correctness of the ExpandingSum expression type and its expansion method."""
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    expression = x + y
    e = cf.ExpandingSum(expression, (x, 1, 3), (y, 1, 3))
    assert int(e.doit()) == 36
    e = cf.ExpandingSum(expression, (x, y, 3), (y, 1, 3))
    assert int(e.doit()) == 24
    e = cf.ExpandingSum(expression, (x, 1, 3), (y, z, 3))
    done_expr = e.doit()
    assert done_expr == e
