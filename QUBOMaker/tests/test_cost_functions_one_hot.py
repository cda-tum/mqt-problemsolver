from __future__ import annotations

from typing import TYPE_CHECKING

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf
from mqt.qubomaker import Graph

if TYPE_CHECKING:
    import sympy as sp

TEST_GRAPH = Graph(
    5,
    [
        (1, 2, 5),
        (1, 3, 4),
        (1, 5, 4),
        (2, 1, 3),
        (2, 4, 3),
        (2, 5, 5),
        (3, 4, 6),
        (3, 5, 2),
        (4, 1, 2),
        (4, 2, 3),
        (4, 5, 4),
        (5, 1, 2),
        (5, 3, 3),
    ],
)


def paths_to_assignment(paths: list[list[int]], n_vertices: int, max_path_length: int) -> dict[sp.Expr, int]:
    result = [
        (cf.X(p + 1, v + 1, i + 1), 1 if len(path) > i and (path[i] == v + 1) else 0)
        for p, path in enumerate(paths)
        for i in range(max_path_length)
        for v in range(n_vertices)
    ]
    print(result)
    return dict(result)


def assert_returns_0(cost_function: pf.CostFunction, path: dict[sp.Expr, int]) -> None:
    settings = pf.PathFindingQUBOGeneratorSettings(pf.EncodingType.ONE_HOT, 1, TEST_GRAPH.n_vertices, False)
    formula = cost_function.get_formula(TEST_GRAPH, settings)
    print(formula)
    value = formula.subs(path).doit()  # type: ignore[no-untyped-call]
    assert value == 0


def assert_returns_gt_0(cost_function: pf.CostFunction, path: dict[sp.Expr, int]) -> None:
    settings = pf.PathFindingQUBOGeneratorSettings(pf.EncodingType.ONE_HOT, 1, TEST_GRAPH.n_vertices, False)
    formula = cost_function.get_formula(TEST_GRAPH, settings)
    value = formula.subs(path).doit()  # type: ignore[no-untyped-call]
    assert value > 0


def test_path_position_is() -> None:
    encoding_A = paths_to_assignment([[1, 3, 4, 2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices)
    encoding_B = paths_to_assignment([[1, 3, 4], [2, 5]], TEST_GRAPH.n_vertices, TEST_GRAPH.n_vertices)

    assert_returns_0(pf.PathPositionIs(2, [3], 1), encoding_A)
    assert_returns_0(pf.PathPositionIs(2, [2, 3], 1), encoding_A)
    assert_returns_0(pf.PathPositionIs(2, [1, 5], 2), encoding_B)

    assert_returns_gt_0(pf.PathPositionIs(1, [4], 1), encoding_A)
    assert_returns_gt_0(pf.PathPositionIs(1, [4, 3, 2], 1), encoding_A)
    assert_returns_gt_0(pf.PathPositionIs(1, [1], 2), encoding_B)


# TODO test remaining ones
