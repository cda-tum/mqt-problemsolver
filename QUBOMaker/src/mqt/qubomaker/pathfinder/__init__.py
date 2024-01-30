"""This module implements the pathfinding functionalities of the QUBOMaker, including, in particular,
all pathfinding-related cost functions and the specialized QUBO generator for pathfinding problems.

Typical usage example:

    ```python
    import mqt.qubomaker.pathfinder as pf

    graph = pf.Graph(5, [(1, 2, 5), (2, 3, 3), (3, 4, 9), (4, 5, 8), (5, 1, 6)])
    settings = pf.PathFindingQUBOGeneratorSettings(pf.EncodingType.ONE_HOT, 1, 5, True)
    generator = pf.PathFindingQUBOGenerator(pf.MinimisePathLength([1]), graph, settings)
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
    MaximisePathLength,
    MinimisePathLength,
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

__all__ = [
    "PathFindingQUBOGenerator",
    "PathFindingQUBOGeneratorSettings",
    "EncodingType",
    "CostFunction",
    "CompositeCostFunction",
    "PathPositionIs",
    "PathStartsAt",
    "PathEndsAt",
    "PathContainsVertices",
    "PathContainsVerticesAtLeastOnce",
    "PathContainsVerticesAtMostOnce",
    "PathContainsVerticesExactlyOnce",
    "PathContainsEdges",
    "PathContainsEdgesExactlyOnce",
    "PathContainsEdgesAtLeastOnce",
    "PathContainsEdgesAtMostOnce",
    "PathBound",
    "PrecedenceConstraint",
    "PathComparison",
    "PathsShareNoVertices",
    "PathsShareNoEdges",
    "PathIsValid",
    "MinimisePathLength",
    "MaximisePathLength",
]
