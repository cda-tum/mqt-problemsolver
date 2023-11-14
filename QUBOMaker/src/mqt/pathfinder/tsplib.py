from tsplib95.models import StandardProblem
import tsplib95
import networkx

from mqt.pathfinder.cost_functions import PathFindingQUBOGenerator, PathFindingQUBOGeneratorSettings
import mqt.pathfinder.cost_functions as cost_functions
import mqt.pathfinder.graph as graph


def to_graph(g: networkx.Graph) -> graph.Graph:
    return graph.Graph(g.number_of_nodes(), g.edges.data("weight"))


def __tsp(
    problem: StandardProblem, encoding_type: cost_functions.EncodingType
) -> PathFindingQUBOGenerator:
    g = to_graph(problem.get_graph())
    settings = PathFindingQUBOGeneratorSettings(encoding_type, 1, g.n_vertices)
    generator = PathFindingQUBOGenerator(
        cost_functions.MinimisePathLength([0], True), g, settings
    )

    generator.add_constraint(cost_functions.PathIsValid([0]))
    generator.add_constraint(cost_functions.PathIsLoop([0]))
    generator.add_constraint(
        cost_functions.PathContainsVerticesExactlyOnce(g.all_vertices, [0])
    )

    return generator


def get_qubo_generator(problem: StandardProblem,
                       encoding_type: cost_functions.EncodingType) -> PathFindingQUBOGenerator:
    match problem.type:
        case "TSP":
            return __tsp(problem, encoding_type)
        case "ATSP":
            raise NotImplementedError()
        case "HCP":
            raise NotImplementedError()
        case "ATSP":
            raise NotImplementedError()
        case "SOP":
            raise NotImplementedError()
        case "CVRP":
            raise NotImplementedError()
        case _:
            raise ValueError("Problem type not supported.")
