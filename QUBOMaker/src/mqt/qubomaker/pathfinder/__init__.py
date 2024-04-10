"""This module implements the pathfinding functionalities of the QUBOMaker.

This, in particular, includes all pathfinding-related cost functions and the specialized QUBO generator for pathfinding problems.

Typical usage example:

    ```python
    import mqt.qubomaker.pathfinder as pf

    graph = pf.Graph(5, [(1, 2, 5), (2, 3, 3), (3, 4, 9), (4, 5, 8), (5, 1, 6)])
    settings = pf.PathFindingQUBOGeneratorSettings(pf.EncodingType.ONE_HOT, 1, 5, True)
    generator = pf.PathFindingQUBOGenerator(pf.MinimizePathLength([1]), graph, settings)
    generator.add_constraint(pf.PathIsValid([1]))
    generator.add_constraint(pf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))

    A = generator_new.construct_qubo_matrix()
    print(A)
    ```
"""

from __future__ import annotations

from .cost_functions import (
    CompositeCostFunction,
    CostFunction,
    EncodingType,
    MaximizePathLength,
    MinimizePathLength,
    PathBound,
    PathComparison,
    PathContainsEdges,
    PathContainsEdgesAtLeastOnce,
    PathContainsEdgesAtMostOnce,
    PathContainsEdgesExactlyOnce,
    PathContainsVertices,
    PathContainsVerticesAtLeastOnce,
    PathContainsVerticesAtMostOnce,
    PathContainsVerticesExactlyOnce,
    PathEndsAt,
    PathIsValid,
    PathPositionIs,
    PathsShareNoEdges,
    PathsShareNoVertices,
    PathStartsAt,
    PrecedenceConstraint,
)
from .pathfinder import PathFindingQUBOGenerator, PathFindingQUBOGeneratorSettings
from .tsplib import from_tsplib_problem

__all__ = [
    "CompositeCostFunction",
    "CostFunction",
    "EncodingType",
    "MaximizePathLength",
    "MinimizePathLength",
    "PathBound",
    "PathComparison",
    "PathContainsEdges",
    "PathContainsEdgesAtLeastOnce",
    "PathContainsEdgesAtMostOnce",
    "PathContainsEdgesExactlyOnce",
    "PathContainsVertices",
    "PathContainsVerticesAtLeastOnce",
    "PathContainsVerticesAtMostOnce",
    "PathContainsVerticesExactlyOnce",
    "PathEndsAt",
    "PathFindingQUBOGenerator",
    "PathFindingQUBOGeneratorSettings",
    "PathIsValid",
    "PathPositionIs",
    "PathStartsAt",
    "PathsShareNoEdges",
    "PathsShareNoVertices",
    "PrecedenceConstraint",
    "from_tsplib_problem",
]
