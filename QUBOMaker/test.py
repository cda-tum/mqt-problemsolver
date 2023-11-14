import cost_functions as cf
import sympy as sp
from graph import Graph

with open("graph", "r") as file:
    graph = Graph.read(file)
encoding_type = cf.EncodingType.ONE_HOT
n_paths = 1
max_path_length = graph.n_vertices
settings = cf.PathFindingQUBOGeneratorSettings(encoding_type, n_paths, max_path_length)
generator = cf.PathFindingQUBOGenerator(cf.MinimisePathLength([1], loop=True), graph, settings)
#generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
print(generator.construct_expansion())