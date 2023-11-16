from __future__ import annotations

from typing import TYPE_CHECKING

import mqt.pathfinder.cost_functions as cost_functions
import mqt.pathfinder.graph as graph
from mqt.pathfinder.cost_functions import PathFindingQUBOGenerator, PathFindingQUBOGeneratorSettings

if TYPE_CHECKING:
    import networkx as nx
    from tsplib95.models import StandardProblem


def to_graph(g: nx.Graph) -> graph.Graph:
    return graph.Graph(g.number_of_nodes(), g.edges.data("weight"))


def __tsp(problem: StandardProblem, encoding_type: cost_functions.EncodingType) -> PathFindingQUBOGenerator:
    g = to_graph(problem.get_graph())
    settings = PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices)
    generator = PathFindingQUBOGenerator(cost_functions.MinimisePathLength([1], True), g, settings)

    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathIsLoop([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))

    return generator


def __hcp(problem: StandardProblem, encoding_type: cost_functions.EncodingType) -> PathFindingQUBOGenerator:
    g = to_graph(problem.get_graph())
    settings = PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices)
    generator = PathFindingQUBOGenerator(None, g, settings)

    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathIsLoop([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))

    return generator


def __sop(problem: StandardProblem, encoding_type: cost_functions.EncodingType) -> PathFindingQUBOGenerator:
    g = to_graph(problem.get_graph())
    settings = PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices)
    generator = PathFindingQUBOGenerator(cost_functions.MinimisePathLength([1]), g, settings)
    generator.add_constraint(cost_functions.PathIsValid([1]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [1]))
    sop_pairs = []
    for u, v, weight in problem.get_graph().edges.data("weight"):
        if weight == -1:
            sop_pairs.append((v, u))
    for u, v in sop_pairs:
        generator.add_constraint(cost_functions.PrecedenceConstraint(u, v, [1]))

    return generator


def get_qubo_generator(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> PathFindingQUBOGenerator:
    match problem.type:
        case "TSP":
            return __tsp(problem, encoding_type)
        case "ATSP":
            return __tsp(problem, encoding_type)
        case "HCP":
            return __hcp(problem, encoding_type)
        case "SOP":
            return __sop(problem, encoding_type)
        case "CVRP":
            msg = "CVRP is not supported as it is not a pure path-finding problem."
            raise NotImplementedError(msg)
        case _:
            msg = "Problem type not supported."
            raise ValueError(msg)
