from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf
from mqt.qubomaker import Graph

if TYPE_CHECKING:
    import sympy as sp


def get_test_graph() -> Graph:
    return Graph(
        5,
        [
            (1, 2, 5),
            (1, 3, 4),
            (1, 5, 4),
            (2, 1, 3),
            (2, 3, 2),
            (2, 4, 3),
            (2, 5, 5),
            (3, 4, 6),
            (3, 5, 2),
            (4, 1, 2),
            (4, 2, 3),
            (4, 5, 4),
            (5, 1, 2),
            (5, 2, 2),
            (5, 3, 3),
            (5, 4, 1),
        ],
    )


def paths_to_assignment(
    paths: list[list[int]], n_vertices: int, max_path_length: int, encoding: pf.EncodingType
) -> dict[sp.Expr, int]:
    if encoding == pf.EncodingType.ONE_HOT:
        return __paths_to_assignment_one_hot(paths, n_vertices, max_path_length)
    if encoding == pf.EncodingType.UNARY:
        return __paths_to_assignment_unary(paths, n_vertices, max_path_length)
    if encoding == pf.EncodingType.BINARY:
        return __paths_to_assignment_binary(paths, n_vertices, max_path_length)
    msg = f"Unknown encoding type: {encoding}"  # type: ignore[unreachable]
    raise ValueError(msg)


def __paths_to_assignment_one_hot(paths: list[list[int]], n_vertices: int, max_path_length: int) -> dict[sp.Expr, int]:
    result = [
        (cf.X(p + 1, v + 1, i + 1), 1 if len(path) > i and (path[i] == v + 1) else 0)
        for p, path in enumerate(paths)
        for i in range(max_path_length)
        for v in range(n_vertices)
    ]
    return dict(result)


def __paths_to_assignment_unary(paths: list[list[int]], n_vertices: int, max_path_length: int) -> dict[sp.Expr, int]:
    result = [
        (cf.X(p + 1, v + 1, i + 1), 1 if len(path) > i and (path[i] >= v + 1) else 0)
        for p, path in enumerate(paths)
        for i in range(max_path_length)
        for v in range(n_vertices)
    ]
    return dict(result)


def __paths_to_assignment_binary(paths: list[list[int]], n_vertices: int, max_path_length: int) -> dict[sp.Expr, int]:
    max_index = int(np.ceil(np.log2(n_vertices + 1)))
    result = [
        (cf.X(p + 1, v + 1, i + 1), 1 if len(path) > i and (((path[i]) >> v) & 1) else 0)
        for p, path in enumerate(paths)
        for i in range(max_path_length)
        for v in range(max_index)
    ]
    return dict(result)


def check_equal(a: pf.PathFindingQUBOGenerator, b: pf.PathFindingQUBOGenerator) -> None:
    assert a.objective_function == b.objective_function
    assert a.graph == b.graph
    assert a.settings == b.settings

    print(len(a.penalties), len(b.penalties))

    for expr, weight in a.penalties:
        assert len([w for (e, w) in b.penalties if e == expr and w == weight]) == 1

    for expr, weight in b.penalties:
        assert len([w for (e, w) in a.penalties if e == expr and w == weight]) == 1
