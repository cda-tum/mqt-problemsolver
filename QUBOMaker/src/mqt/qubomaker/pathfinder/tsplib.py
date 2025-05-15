"""Provides support for the TSPLib format as input for the pathfinding QUBOMaker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mqt.qubomaker
import mqt.qubomaker.pathfinder
from mqt.qubomaker.pathfinder import cost_functions

if TYPE_CHECKING:
    import networkx as nx
    from tsplib95.models import StandardProblem


def __check_forced_edges(problem: StandardProblem) -> cost_functions.CostFunction | None:
    if not problem.fixed_edges:
        return None
    forced_edges: list[tuple[int, int]] = []
    for i, j in problem.fixed_edges:
        forced_edges.append((i + 1, j + 1))
    return cost_functions.PathContainsEdgesExactlyOnce(forced_edges, [1])


def to_graph(g: nx.Graph) -> mqt.qubomaker.Graph:
    """Transforms a networkx graph into a Graph object.

    Args:
        g (nx.Graph): The networkx graph to be transformed.

    Returns:
        Graph: The transformed graph.
    """
    return mqt.qubomaker.Graph(g.number_of_nodes(), g.edges.data("weight"))


def __tsp(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> mqt.qubomaker.pathfinder.PathFindingQUBOGenerator:
    """Constructs a QUBO generator for a TSP problem.

    Args:
        problem (StandardProblem): The TSP problem.
        encoding_type (cost_functions.EncodingType): The desired encoding type.

    Returns:
        PathFindingQUBOGenerator: The constructed QUBO generator.
    """
    g = to_graph(problem.get_graph())
    settings = mqt.qubomaker.pathfinder.PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices, True)
    generator = mqt.qubomaker.pathfinder.PathFindingQUBOGenerator(cost_functions.MinimizePathLength([1]), g, settings)

    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))

    generator.add_constraint_if_exists(__check_forced_edges(problem))

    return generator


def __hcp(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> mqt.qubomaker.pathfinder.PathFindingQUBOGenerator:
    """Constructs a QUBO generator for a HCP problem.

    Args:
        problem (StandardProblem): The HCP problem.
        encoding_type (cost_functions.EncodingType): The desired encoding type.

    Returns:
        PathFindingQUBOGenerator: The constructed QUBO generator.
    """
    g = to_graph(problem.get_graph())
    settings = mqt.qubomaker.pathfinder.PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices, True)
    generator = mqt.qubomaker.pathfinder.PathFindingQUBOGenerator(None, g, settings)

    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))

    generator.add_constraint_if_exists(__check_forced_edges(problem))

    return generator


def __sop(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> mqt.qubomaker.pathfinder.PathFindingQUBOGenerator:
    """Constructs a QUBO generator for a SOP problem.

    Args:
        problem (StandardProblem): The SOP problem.
        encoding_type (cost_functions.EncodingType): The desired encoding type.

    Returns:
        PathFindingQUBOGenerator: The constructed QUBO generator.
    """
    g = to_graph(problem.get_graph())
    settings = mqt.qubomaker.pathfinder.PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices, False)
    generator = mqt.qubomaker.pathfinder.PathFindingQUBOGenerator(cost_functions.MinimizePathLength([1]), g, settings)
    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))
    sop_pairs = []
    for u, v, weight in problem.get_graph().edges.data("weight"):
        if weight == -1:
            sop_pairs.append((v + 1, u + 1))
    for u, v in sop_pairs:
        generator.add_constraint(cost_functions.PrecedenceConstraint(u, v, [1]))

    generator.add_constraint_if_exists(__check_forced_edges(problem))

    return generator


def from_tsplib_problem(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> mqt.qubomaker.pathfinder.PathFindingQUBOGenerator:
    """Constructs a QUBO generator for a given problem in TSPLib format.

    Args:
        problem (StandardProblem): The TSPLib problem.
        encoding_type (cost_functions.EncodingType): The desired encoding type.

    Raises:
        NotImplementedError: If a CVRP problem is given, as this problem type cannot be solved by the pathfinder.
        ValueError: If an unknown problem type is given.

    Returns:
        PathFindingQUBOGenerator: The constructed QUBO generator.
    """
    if problem.type in {"TSP", "ATSP"}:
        return __tsp(problem, encoding_type)
    if problem.type == "HCP":
        return __hcp(problem, encoding_type)
    if problem.type == "SOP":
        return __sop(problem, encoding_type)
    if problem.type == "CVRP":
        msg = "CVRP is not supported as it is not a pure path-finding problem."
        raise ValueError(msg)
    msg = "Problem type not supported."
    raise ValueError(msg)
