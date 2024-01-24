from __future__ import annotations

from typing import cast

import pytest
import sympy as sp

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf

from .utils_test import get_test_graph, paths_to_assignment

TEST_GRAPH = get_test_graph()


def evaluate(
    cost_function: pf.CostFunction, path: dict[sp.Expr, int], encoding: pf.EncodingType, loop: bool, n_paths: int = 1
) -> int:
    settings = pf.PathFindingQUBOGeneratorSettings(encoding, n_paths, TEST_GRAPH.n_vertices, loop)
    formula = cost_function.get_formula(TEST_GRAPH, settings)
    assignment = [
        (cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, TEST_GRAPH.n_vertices + 1, i + 1), 0)
        for p in range(settings.n_paths)
        for i in range(settings.max_path_length + 1)
    ]  # x_{p, |V| + 1, i} = 0 for all p, i
    assignment += [
        (
            cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, settings.max_path_length + 1),
            cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, 1)
            if settings.loops
            else sp.Integer(0)
            if settings.encoding_type != pf.EncodingType.BINARY
            else cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, settings.max_path_length),
        )
        for p in range(settings.n_paths)
        for v in range(TEST_GRAPH.n_vertices)
    ]  # x_{p, v, N + 1} = x_{p, v, 1} for all p, v if loop, otherwise 0
    assignment += [
        (cf._FormulaHelpers.adjacency(i + 1, j + 1), TEST_GRAPH.adjacency_matrix[i, j])
        for i in range(TEST_GRAPH.n_vertices)
        for j in range(TEST_GRAPH.n_vertices)
    ]
    return cast(
        int,
        formula.subs(path)  # type: ignore[no-untyped-call]
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
        (cf.EncodingType.UNARY, False),
        (cf.EncodingType.BINARY, False),
        (cf.EncodingType.ONE_HOT, True),
        (cf.EncodingType.UNARY, True),
        (cf.EncodingType.BINARY, True),
    ],
)
class TestCostFunctions:
    def test_path_position_is(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathPositionIs(2, [3], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathPositionIs(2, [2, 3], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathPositionIs(2, [1, 5], 2), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PathPositionIs(1, [4], 1), encoding_A, encoding_type, loop) > 0
        assert evaluate(pf.PathPositionIs(1, [4, 3, 2], 1), encoding_A, encoding_type, loop) > 0
        assert evaluate(pf.PathPositionIs(1, [1], 2), encoding_B, encoding_type, loop, n_paths=2) > 0

    def test_path_starts_at_ends_at(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathStartsAt([1, 4], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathStartsAt([1], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathStartsAt([1], 1), encoding_B, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PathStartsAt([2], 2), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PathEndsAt([3, 5], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathEndsAt([5], 1), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathEndsAt([4], 1), encoding_B, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PathEndsAt([5], 2), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PathStartsAt([4, 3, 5], 1), encoding_A, encoding_type, loop) > 0
        assert evaluate(pf.PathStartsAt([2], 1), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PathStartsAt([5], 2), encoding_B, encoding_type, loop, n_paths=2) > 0

        assert evaluate(pf.PathEndsAt([1, 2, 3, 4], 1), encoding_A, encoding_type, loop) > 0
        assert evaluate(pf.PathEndsAt([5], 1), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PathEndsAt([4], 2), encoding_B, encoding_type, loop, n_paths=2) > 0

    def test_path_contains_vertices_exactly_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathContainsVerticesExactlyOnce([1, 4], [1]), encoding_A, encoding_type, loop) == 0
        assert (
            evaluate(pf.PathContainsVerticesExactlyOnce([3], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0
        )

        assert evaluate(pf.PathContainsVerticesExactlyOnce([4], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PathContainsVerticesExactlyOnce([5], [1]), encoding_A, encoding_type, loop) > 0

    def test_path_contains_vertices_at_least_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathContainsVerticesAtLeastOnce([1, 4], [1]), encoding_A, encoding_type, loop) == 0
        assert (
            evaluate(pf.PathContainsVerticesAtLeastOnce([3], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0
        )
        assert evaluate(pf.PathContainsVerticesAtLeastOnce([5], [1]), encoding_A, encoding_type, loop) == 0

        assert evaluate(pf.PathContainsVerticesAtLeastOnce([4], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) > 0

    def test_path_contains_vertices_at_most_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathContainsVerticesAtMostOnce([1, 4], [1]), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathContainsVerticesAtMostOnce([3], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PathContainsVerticesAtMostOnce([4], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PathContainsVerticesAtMostOnce([5], [1]), encoding_A, encoding_type, loop) > 0

    def test_path_contains_edges_exactly_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathContainsEdgesExactlyOnce([(5, 4), (4, 1)], [1]), encoding_A, encoding_type, loop) == 0
        assert (
            evaluate(pf.PathContainsEdgesExactlyOnce([(1, 3), (3, 4)], [1]), encoding_B, encoding_type, loop, n_paths=2)
            == 0
        )

        assert (
            evaluate(pf.PathContainsEdgesExactlyOnce([(1, 3)], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) > 0
        )
        assert evaluate(pf.PathContainsEdgesExactlyOnce([(1, 5)], [1]), encoding_A, encoding_type, loop) > 0

    def test_path_contains_edges_at_least_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathContainsEdgesAtLeastOnce([(5, 4), (4, 1)], [1]), encoding_A, encoding_type, loop) == 0
        assert (
            evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 3), (3, 4)], [1]), encoding_B, encoding_type, loop, n_paths=2)
            == 0
        )
        assert evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 5)], [1]), encoding_A, encoding_type, loop) == 0

        assert (
            evaluate(pf.PathContainsEdgesAtLeastOnce([(1, 3)], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) > 0
        )

    def test_path_contains_edges_at_most_once(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 5, 4, 1, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert (
            evaluate(pf.PathContainsEdgesAtMostOnce([(5, 4), (4, 1), (5, 5)], [1]), encoding_A, encoding_type, loop)
            == 0
        )
        assert (
            evaluate(pf.PathContainsEdgesAtMostOnce([(1, 3), (3, 4)], [1]), encoding_B, encoding_type, loop, n_paths=2)
            == 0
        )
        assert (
            evaluate(pf.PathContainsEdgesAtMostOnce([(1, 3)], [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0
        )

        assert evaluate(pf.PathContainsEdgesAtMostOnce([(1, 5)], [1]), encoding_A, encoding_type, loop) > 0

    def test_precedence_constraint(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PrecedenceConstraint(1, 3, [1]), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PrecedenceConstraint(3, 4, [1]), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PrecedenceConstraint(3, 2, [1]), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PrecedenceConstraint(4, 2, [1]), encoding_B, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PrecedenceConstraint(1, 4, [1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PrecedenceConstraint(2, 4, [1]), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PrecedenceConstraint(5, 2, [2]), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PrecedenceConstraint(4, 3, [1]), encoding_A, encoding_type, loop) > 0
        assert evaluate(pf.PrecedenceConstraint(5, 1, [1]), encoding_A, encoding_type, loop) > 0

    def test_share_no_vertices(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 5], [1, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathsShareNoVertices(1, 2), encoding_A, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PathsShareNoVertices(1, 2), encoding_B, encoding_type, loop, n_paths=3) == 0
        assert evaluate(pf.PathsShareNoVertices(2, 3), encoding_B, encoding_type, loop, n_paths=3) == 0

        assert evaluate(pf.PathsShareNoVertices(1, 3), encoding_B, encoding_type, loop, n_paths=3) > 0

    def test_share_no_edges(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 3], [1, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathsShareNoEdges(1, 2), encoding_A, encoding_type, loop, n_paths=2) == 0
        assert evaluate(pf.PathsShareNoEdges(1, 2), encoding_B, encoding_type, loop, n_paths=3) == 0
        assert evaluate(pf.PathsShareNoEdges(2, 3), encoding_B, encoding_type, loop, n_paths=3) == 0

        assert evaluate(pf.PathsShareNoEdges(1, 3), encoding_B, encoding_type, loop, n_paths=3) > 0

    def test_maximize_minimize(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.MinimisePathLength([1]), encoding_A, encoding_type, loop) == 18 if not loop else 20
        assert (
            evaluate(pf.MinimisePathLength([1]), encoding_B, encoding_type, loop, n_paths=2) == 10 if not loop else 12
        )
        assert evaluate(pf.MinimisePathLength([2]), encoding_B, encoding_type, loop, n_paths=2) == 5 if not loop else 7
        assert (
            evaluate(pf.MinimisePathLength([1, 2]), encoding_B, encoding_type, loop, n_paths=2) == 15
            if not loop
            else 19
        )

        assert evaluate(pf.MaximisePathLength([1]), encoding_A, encoding_type, loop) == -18 if not loop else -20
        assert (
            evaluate(pf.MaximisePathLength([1]), encoding_B, encoding_type, loop, n_paths=2) == -10 if not loop else -12
        )
        assert (
            evaluate(pf.MaximisePathLength([2]), encoding_B, encoding_type, loop, n_paths=2) == -5 if not loop else -7
        )
        assert (
            evaluate(pf.MaximisePathLength([1, 2]), encoding_B, encoding_type, loop, n_paths=2) == -15
            if not loop
            else -19
        )

    def test_is_valid(self, encoding_type: pf.EncodingType, loop: bool) -> None:
        encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type)
        encoding_B = paths_to_assignment(
            [[1, 3, 4], [4, 3]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices, encoding_type
        )

        assert evaluate(pf.PathIsValid([1]), encoding_A, encoding_type, loop) == 0
        assert evaluate(pf.PathIsValid([1]), encoding_B, encoding_type, loop, n_paths=2) == 0

        assert evaluate(pf.PathIsValid([2]), encoding_B, encoding_type, loop, n_paths=2) > 0
        assert evaluate(pf.PathIsValid([1, 2]), encoding_B, encoding_type, loop, n_paths=2) > 0
