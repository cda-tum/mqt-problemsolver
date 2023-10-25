import functools
from enum import Enum
from typing import Any, Callable, Self
import arithmetic
from arithmetic import ArithmeticItem, AssigningTransformer, SimplifyingTransformer, Variable
from graph import Graph
from qubo_generator import QUBOGenerator
import numpy as np


class EncodingType(Enum):
    OneHot = 1,
    Unary = 2,
    Binary = 3
    


class CostFunction:
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        return arithmetic.Constant(0)



class CompositeCostFunction(CostFunction):
    summands: list[tuple[CostFunction, int]]
    
    def __init__(self, *parts: tuple[CostFunction, int]) -> None:
        super().__init__()
        self.summands = list(parts)
        
    def __str__(self) -> str:
        return "   " + "\n + ".join([f"{w} * {fn.__class__.__name__}" for (fn, w) in self.summands])
    
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        return functools.reduce(lambda a, b: arithmetic.Addition(
                                                a, arithmetic.Multiplication(
                                                    arithmetic.Constant(b[1]),
                                                    b[0].get_formula(graph, encoding)
                                                )),
                                self.summands[1:],
                                arithmetic.Multiplication(
                                    arithmetic.Constant(self.summands[0][1]),
                                    self.summands[0][0].get_formula(graph, encoding)
                                ))
        


class PathPositionIs(CostFunction):
    vertex_ids: list[int]
    position: int
    
    def __init__(self, position: int, vertex_ids: list[int]) -> None:
        super().__init__()
        self.vertex_ids = vertex_ids
        self.position = position
        
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        if len(self.vertex_ids) == 1:
            return self.__get_formula_single(graph, encoding, self.vertex_ids[0])
        return (1 - arithmetic.SumSet(arithmetic.Variable("x", ["v", 1], []), 
                                      [arithmetic.Variable("v")], 
                                      f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                                      lambda: [arithmetic.Constant(v) for v in self.vertex_ids]
                                      )) ** 2
    
    def __get_formula_single(self, graph: Graph, encoding: EncodingType, v: int) -> arithmetic.ArithmeticItem:
        pos = self.position if self.position > 0 else graph.n_vertices
        return (1 - arithmetic.Variable("x", [v, pos], []))
        


class PathStartsAt(PathPositionIs):
    vertex_ids: list[int]
    
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(1, vertex_ids)
        


class PathEndsAt(PathPositionIs):
    vertex_ids: list[int]
    
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(-1, vertex_ids)
        


class PathContainsVertices(CostFunction):
    vertex_ids: list[int]
    min_occurrences: int
    max_occurrences: int
    
    def __init__(self, min_occurrences: int, max_occurrences: int, vertex_ids: list[int]) -> None:
        super().__init__()
        self.vertex_ids = vertex_ids
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        


class PathContainsVerticesExactlyOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(1, 1, vertex_ids)
        
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        if len(self.vertex_ids) == 1:
            return self.__get_formula_single(graph, encoding, self.vertex_ids[0])
        return arithmetic.SumSet(
            (1 - arithmetic.SumFromTo(arithmetic.Variable("x", ["v", "i"], []), arithmetic.Variable("i"), arithmetic.Constant(1), arithmetic.Constant(graph.n_vertices))) ** 2,
            [arithmetic.Variable("v")],
            f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
            lambda: [arithmetic.Constant(v) for v in self.vertex_ids]
        )
    
    def __get_formula_single(self, graph: Graph, encoding: EncodingType, v: int) -> arithmetic.ArithmeticItem:
        return (1 - arithmetic.SumFromTo(arithmetic.Variable("x", [v, "i"], []), arithmetic.Variable("i"), arithmetic.Constant(1), arithmetic.Constant(graph.n_vertices))) ** 2


class PathContainsVerticesAtLeastOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(1, -1, vertex_ids)



class PathContainsVerticesAtMostOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(0, 1, vertex_ids)
        


class PathIsLoop(CostFunction):
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        return arithmetic.SumSet(
            arithmetic.Multiplication(arithmetic.Variable("x", ["v", graph.n_vertices]), arithmetic.Variable("x", ["w", 1])), 
            [arithmetic.Variable("v"), arithmetic.Variable("w")],
            "\\not\\in E",
            lambda: [(arithmetic.Constant(i + 1), arithmetic.Constant(j + 1)) for i in range(graph.n_vertices) for j in range(graph.n_vertices) if graph.adjacency_matrix[i, j] == 0]
        )



class PathComparison(CostFunction):
    path_ids: list[int]
    
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__()
        self.path_ids = path_ids
        


class PathsShareNoVertices(PathComparison):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
        


class PathsShareNoEdges(PathComparison):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
        


class PathIsValid(PathComparison):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
        
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        return arithmetic.SumSet(
            arithmetic.Addition(
                arithmetic.SumFromTo(
                    arithmetic.Multiplication(arithmetic.Variable("x", ["v", "i"]), arithmetic.Variable("x", ["w", arithmetic.Variable("i") + 1])), 
                    arithmetic.Variable("i"), 
                    arithmetic.Constant(1), 
                    arithmetic.Constant(graph.n_vertices - 1)
                ),
                arithmetic.SumFromTo(
                    (
                        1 - arithmetic.SumSet(
                            arithmetic.Variable("x", ["v", "i"]),
                            [arithmetic.Variable("v")],
                            r" \in V",
                            lambda: [arithmetic.Constant(i) for i in graph.all_vertices]
                        )
                    )**2,
                    arithmetic.Variable("i"), 
                    arithmetic.Constant(1), 
                    arithmetic.Constant(graph.n_vertices)
                ),
            ),
            [arithmetic.Variable("v"), arithmetic.Variable("w")],
            "\\not\\in E",
            lambda: [(arithmetic.Constant(i + 1), arithmetic.Constant(j + 1)) for i in range(graph.n_vertices) for j in range(graph.n_vertices) if graph.adjacency_matrix[i, j] == 0]
        )
        


class MinimisePathLength(PathComparison):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
    
    def get_formula(self, graph: Graph, encoding: EncodingType) -> arithmetic.ArithmeticItem:
        return arithmetic.SumFromTo(
            arithmetic.SumFromTo(
                arithmetic.SumFromTo(
                    arithmetic.Variable("A", ["v", "w"]) * arithmetic.Variable("x", ["v", "i"]) * arithmetic.Variable("x", ["w", arithmetic.Variable("i") + 1]),
                    arithmetic.Variable("w"),
                    arithmetic.Constant(1),
                    arithmetic.Constant(graph.n_vertices)
                ),
                arithmetic.Variable("v"),
                arithmetic.Constant(1),
                arithmetic.Constant(graph.n_vertices)
            ),
            arithmetic.Variable("i"),
            arithmetic.Constant(1),
            arithmetic.Constant(graph.n_vertices - 1)
        )
        
def merge(cost_functions: list[CostFunction], optimisation_goals: list[CostFunction]) -> CompositeCostFunction:
    return CompositeCostFunction(*([(f, 1) for f in cost_functions] + [(f, 1) for f in optimisation_goals]))



class PathFindingQUBOGenerator(QUBOGenerator):
    encoding_type: EncodingType
    graph: Graph
    n_paths: int
    max_path_length: int
    
    def __init__(self, objective_function: CostFunction,
                 encoding_type: EncodingType,
                 graph: Graph,
                 n_paths: int,
                 max_path_length: int) -> None:
        super().__init__(objective_function.get_formula(graph, encoding_type))
        self.encoding_type = encoding_type
        self.graph = graph
        self.n_paths = n_paths
        self.max_path_length = max_path_length
        
    def add_constraint(self, constraint: CostFunction) -> Self:
        self.add_penalty(constraint.get_formula(self.graph, self.encoding_type), self.__optimal_lambda())
        return self
    
    def __optimal_lambda(self) -> int:
        return np.sum(self.graph.adjacency_matrix)
    
    def construct_expansion(self) -> ArithmeticItem:
        s = super().construct_expansion()
        assignment = [(arithmetic.Variable("A", [i + 1, j + 1]), arithmetic.Constant(self.graph.adjacency_matrix[i, j])) for i in range(self.graph.n_vertices) for j in range(self.graph.n_vertices)]
        return SimplifyingTransformer().transform( 
            AssigningTransformer(*assignment).transform(s)
        )
        
    def get_qubit_count(self) -> int:
        return self.n_paths * self.max_path_length * self.graph.n_vertices #TODO
    
    def get_variable_index(self, variable: Variable) -> int:
        if not variable.name == "x":
            raise ValueError()
        if len(variable.subscripts) != 2:
            raise ValueError()
        if variable.superscripts:
            raise ValueError()
        
        if not isinstance(variable.subscripts[0], arithmetic.Constant):
            raise ValueError()
        
        if not isinstance(variable.subscripts[1], arithmetic.Constant):
            raise ValueError()
        
        return int((variable.subscripts[0].value - 1) + (variable.subscripts[1].value - 1) * self.graph.n_vertices + 1)
    
    def decode_bit_array(self, array: list[int]) -> Any:
        path = []
        for i, bit in enumerate(array):
            if bit == 0:
                continue
            v = i % self.graph.n_vertices
            s = i // self.graph.n_vertices
            path.append((v, s))
        path.sort(key=lambda x: x[1])
        path = [entry[0] + 1 for entry in path]
        return path