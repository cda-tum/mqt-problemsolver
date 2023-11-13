from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from enum import Enum
from typing import Any, Self
import numpy as np

import arithmetic
from arithmetic import ArithmeticItem, AssigningTransformer, Constant, \
    SimplifyingTransformer, Variable
from graph import Graph
from qubo_generator import QUBOGenerator


class EncodingType(Enum):
    ONE_HOT = 1
    UNARY = 2
    BINARY = 3

def get_encoding_variable_one_hot(path: any, vertex: any, position: any) -> Variable:
    return Variable("x", [path, vertex, position])

def get_encoding_sum_unary(path: any, vertex: any, n: int) -> arithmetic.SumFromTo:
    raise NotImplementedError()


#pylint: disable=too-few-public-methods
class CostFunction(ABC):
    def get_formula(self, graph: Graph, encoding: EncodingType) -> ArithmeticItem:
        if encoding == EncodingType.ONE_HOT:
            return self.get_formula_one_hot(graph)
    
    @abstractmethod
    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        pass
    
    @abstractmethod
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


#pylint: disable=too-few-public-methods
class CompositeCostFunction(CostFunction):
    summands: list[tuple[CostFunction, int]]

    def __init__(self, *parts: tuple[CostFunction, int]) -> None:
        self.summands = list(parts)

    def __str__(self) -> str:
        return "   " + "\n + ".join([f"{w} * {fn}" for (fn, w) in self.summands])

    def get_formula(self, graph: Graph, encoding: EncodingType) -> ArithmeticItem:
        return functools.reduce(lambda a, b: a + b[1] * b[0].get_formula(graph, encoding),
                                self.summands[1:],
                                self.summands[0][1] * self.summands[0][0].get_formula(graph, encoding)
                            )


#pylint: disable=too-few-public-methods
class PathPositionIs(CostFunction):
    vertex_ids: list[int]
    path: int
    position: int

    def __init__(self, position: int, vertex_ids: list[int], path: int) -> None:
        self.vertex_ids = vertex_ids
        self.position = position
        self.path = path

    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        if len(self.vertex_ids) == 1:
            return self.__get_formula_one_hot_single(graph, self.vertex_ids[0])
        return arithmetic.SumSet(1 - get_encoding_variable_one_hot(self.path, "v", 1),
                [Variable("v")],
                f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                lambda: [Constant(v) for v in self.vertex_ids]
                ) ** 2

    def __get_formula_one_hot_single(self, graph: Graph, v: int) -> ArithmeticItem:
        pos = self.position if self.position > 0 else graph.n_vertices
        return 1 - get_encoding_variable_one_hot(self.path, v, pos)
    

    def __str__(self) -> str:
        return f"PathPosition[{self.position}]Is[{','.join([str(v) for v in self.vertex_ids])}]"
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()



#pylint: disable=too-few-public-methods
class PathStartsAt(PathPositionIs):
    vertex_ids: list[int]

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        super().__init__(1, vertex_ids, path)

    def __str__(self) -> str:
        return f"PathStartsAt[{','.join([str(v) for v in self.vertex_ids])}]"



#pylint: disable=too-few-public-methods
class PathEndsAt(PathPositionIs):
    vertex_ids: list[int]

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        super().__init__(-1, vertex_ids, path)

    def __str__(self) -> str:
        return f"PathEndsAt[{','.join([str(v) for v in self.vertex_ids])}]"



#pylint: disable=too-few-public-methods
class PathContainsVertices(CostFunction):
    vertex_ids: list[int]
    min_occurrences: int
    max_occurrences: int
    possible_paths: list[int]

    def __init__(self, min_occurrences: int, max_occurrences: int, vertex_ids: list[int], possible_paths: list[int]) -> None:
        self.vertex_ids = vertex_ids
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.possible_paths = possible_paths

    def __str__(self) -> str:
        vertices = ','.join([str(v) for v in self.vertex_ids])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"



#pylint: disable=too-few-public-methods
class PathContainsVerticesExactlyOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], possible_paths: list[int]) -> None:
        super().__init__(1, 1, vertex_ids, possible_paths)

    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        if len(self.vertex_ids) == 1:
            return self.__get_formula_one_hot_single(graph, self.vertex_ids[0])
        return arithmetic.SumSet( 
                arithmetic.SumSet(
                    (1 - arithmetic.SumFromTo(
                        get_encoding_variable_one_hot("p" , "v", "i"),
                        Variable("i"),
                        Constant(1),
                        Constant(graph.n_vertices))
                    ) ** 2,
                    [Variable("v")],
                    f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                    lambda: [Constant(v) for v in self.vertex_ids]),
                [Variable("p")],
                rf"\in \left\{{{','.join([str(p) for p in self.possible_paths])}\right\}}",
                lambda: [Constant(p) for p in self.possible_paths]
            )

    def __get_formula_one_hot_single(self, graph: Graph, v: int) -> ArithmeticItem:
        return (1 - arithmetic.SumSet(
                arithmetic.SumFromTo(
                    Variable(get_encoding_variable_one_hot("p", v, "i")),
                    Variable("i"),
                    Constant(1),
                    Constant(graph.n_vertices)
                ),
                [Variable("p")],
                rf"\in \left\{{{', '.join([str(p) for p in self.possible_paths])} \right\}} ",
                lambda: [Constant(i) for i in self.possible_paths],
        )) ** 2
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathContainsVerticesAtLeastOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(1, -1, vertex_ids)

    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathContainsVerticesAtMostOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int]) -> None:
        super().__init__(0, 1, vertex_ids)

    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathBound(CostFunction):
    path_ids: list[int]

    def __init__(self, path_ids: list[int]) -> None:
        self.path_ids = path_ids

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{','.join([str(id) for id in self.path_ids])}]"


#pylint: disable=too-few-public-methods
class PathIsLoop(PathBound):
    
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
    
    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        return arithmetic.SumSet(
            arithmetic.SumSet(
                get_encoding_variable_one_hot("p", "v", graph.n_vertices) * get_encoding_variable_one_hot("p", "w", 1),
                [Variable("v"), Variable("w")],
                "\\not\\in E",
                
                lambda: [
                    (Constant(i + 1), Constant(j + 1))
                    for i in range(graph.n_vertices)
                    for j in range(graph.n_vertices)
                    if graph.adjacency_matrix[i, j] == 0
                ]),
            [Variable("p")],
            rf"\in \left\{{{', '.join([str(p) for p in self.path_ids])}\right \}}",
            lambda: [Constant(i) for i in self.path_ids]
        )
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathsShareNoVertices(PathBound):
    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()

    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathsShareNoEdges(PathBound):
    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()
    
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()


#pylint: disable=too-few-public-methods
class PathIsValid(PathBound):

    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)
        
    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        return arithmetic.SumSet(
            arithmetic.SumSet(
                arithmetic.SumFromTo(
                    get_encoding_variable_one_hot("p", "v", "i") * get_encoding_variable_one_hot("p", "w", Variable("i") + 1),
                    Variable("i"),
                    Constant(1),
                    Constant(graph.n_vertices - 1)
                ) + 
                arithmetic.SumFromTo(
                    (
                        1 - arithmetic.SumSet(
                            get_encoding_variable_one_hot("p", "v", "i"),
                            [Variable("v")],
                            r" \in V",
                            lambda: [Constant(i) for i in graph.all_vertices]
                        )
                    )**2,
                    Variable("i"),
                    Constant(1),
                    Constant(graph.n_vertices)
                ),
                [Variable("v"), Variable("w")],
                "\\not\\in E",
                lambda: [
                    (Constant(i + 1), Constant(j + 1))
                    for i in range(graph.n_vertices)
                    for j in range(graph.n_vertices)
                    if graph.adjacency_matrix[i, j] == 0]
            ),
            [Variable("p")],
            f"\\in \\left\\{{ {', '.join([str(v) for v in self.path_ids])} \\right\\}}",
            lambda: [Constant(i) for i in self.path_ids] 
        )
        
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()
        



#pylint: disable=too-few-public-methods
class MinimisePathLength(PathBound):
    
    loop: bool
    
    def __init__(self, path_ids: list[int], loop: bool = False) -> None:
        super().__init__(path_ids)
        self.loop = loop

    def get_formula_one_hot(self, graph: Graph) -> ArithmeticItem:
        return arithmetic.SumSet(
                arithmetic.SumFromTo(
                    arithmetic.SumFromTo(
                        get_encoding_variable_one_hot("p", "v", graph.n_vertices) *
                        get_encoding_variable_one_hot("p", "w", 1) * (1 if self.loop else 0) + 
                            arithmetic.SumFromTo(
                                Variable("A", ["v", "w"]) * get_encoding_variable_one_hot("p", "v", "i")
                                    * get_encoding_variable_one_hot("p", "w", Variable("i") + 1),
                                    Variable("i"),
                                Constant(1),
                                Constant(graph.n_vertices - 1)),
                        Variable("w"),
                        Constant(1),
                        Constant(graph.n_vertices)
                    ),
                    Variable("v"),
                    Constant(1),
                    Constant(graph.n_vertices)
                ),
            [Variable("p")],
            rf"\in \left\{{{', '.join([str(p) for p in self.path_ids])}\right \}}",
            lambda: [Constant(i) for i in self.path_ids]
        )
        
    def get_formula_unary(self, graph: Graph) -> ArithmeticItem:
        raise NotImplementedError()

def merge(cost_functions: list[CostFunction],
          optimisation_goals: list[CostFunction]) -> CompositeCostFunction:
    return CompositeCostFunction(*(
        [(f, 1) for f in cost_functions]
        + [(f, 1) for f in optimisation_goals]
    ))

@dataclass
class PathFindingQUBOGeneratorSettings:
    encoding_type: EncodingType
    n_paths: int
    max_path_length: int

class PathFindingQUBOGenerator(QUBOGenerator):
    graph: Graph
    settings: PathFindingQUBOGeneratorSettings

    def __init__(self, objective_function: CostFunction,
                 graph: Graph,
                 settings: PathFindingQUBOGeneratorSettings) -> None:
        super().__init__(objective_function.get_formula(graph, settings.encoding_type))
        self.graph = graph
        self.settings = settings

    def add_constraint(self, constraint: CostFunction) -> Self:
        self.add_penalty(constraint.get_formula(self.graph, self.settings.encoding_type))
        return self

    def _select_lambdas(self) -> list[tuple[ArithmeticItem, int]]:
        return [(expr, lam if lam else self.__optimal_lambda()) for (expr, lam) in self.penalties]

    def __optimal_lambda(self) -> int:
        return np.sum(self.graph.adjacency_matrix)

    def _construct_expansion(self, expression) -> ArithmeticItem:
        assignment = [
            (Variable("A", [i + 1, j + 1]), Constant(self.graph.adjacency_matrix[i, j]))
            for i in range(self.graph.n_vertices)
            for j in range(self.graph.n_vertices)
        ]
        return SimplifyingTransformer().transform(
            AssigningTransformer(*assignment).transform(expression)
        )

    def get_variable_index(self, variable: Variable) -> int:
        if not variable.name == "x":
            raise ValueError()
        if len(variable.subscripts) < 2:
            raise ValueError()
        if variable.superscripts:
            raise ValueError()
        if not isinstance(variable.subscripts[0], Constant):
            raise ValueError()
        if not isinstance(variable.subscripts[1], Constant):
            raise ValueError()

        return int(
            (variable.subscripts[1].value - 1)
            + (variable.subscripts[2].value - 1) * self.settings.max_path_length + 1)

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

    def _get_all_variables(self) -> list[tuple[str, int]]:
        result = []
        for v in self.graph.all_vertices:
            for i in range(0, self.settings.max_path_length):
                var = get_encoding_variable_one_hot(1, v, i + 1) #TODO
                result.append((str(var), self.get_variable_index(var)))
        return result
