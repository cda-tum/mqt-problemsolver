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
    generator = PathFindingQUBOGenerator(cost_functions.MinimisePathLength([0], True), g, settings)

    generator.add_constraint(cost_functions.PathIsValid([0]))
    generator.add_constraint(cost_functions.PathIsLoop([0]))
    generator.add_constraint(cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [0]))

    return generator


def get_qubo_generator(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> PathFindingQUBOGenerator:
    match problem.type:
        case "TSP":
            return __tsp(problem, encoding_type)
        case "ATSP":
            raise NotImplementedError
        case "HCP":
            raise NotImplementedError
        case "ATSP":
            raise NotImplementedError
        case "SOP":
            raise NotImplementedError
        case "CVRP":
            raise NotImplementedError
        case _:
            msg = "Problem type not supported."
            raise ValueError(msg)
