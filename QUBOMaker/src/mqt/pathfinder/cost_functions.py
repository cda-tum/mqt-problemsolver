from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Self, Sequence, cast

import numpy as np
import sympy as sp

from mqt.pathfinder.qubo_generator import QUBOGenerator

if TYPE_CHECKING:
    from mqt.pathfinder.graph import Graph

# Python 3.12
# type SetCallback = Callable[[], list[sp.Expr | int | float | tuple[sp.Expr | int | float, ...]]]
SetCallback = Callable[[], list[sp.Expr | int | float | tuple[sp.Expr | int | float, ...]]]
GetVariableFunction = Callable[[Any, Any, Any], sp.Expr]


class EncodingType(Enum):
    ONE_HOT = 1
    UNARY = 2
    BINARY = 3


class A(sp.Function):
    def _latex(self, _printer: sp.StrPrinter, *_args: Any, **_kwargs: Any) -> str:
        v, w = (self.args[0], self.args[1])
        return rf"A_{{{v},{w}}}"


class X(sp.Function):
    def _latex(self, _printer: sp.StrPrinter, *_args: Any, **_kwargs: Any) -> str:
        p, v, i = (self.args[0], self.args[1], self.args[2])
        return rf"x_{{{p},{v},{i}}}"


class _FormulaHelpers:
    @staticmethod
    def sum_from_to(expression: sp.Expr, var: str, from_number: int, to_number: int) -> sp.Expr:
        s = sp.Symbol(var)  # type: ignore[no-untyped-call]
        return sp.Sum(expression, (s, from_number, to_number))  # type: ignore[no-untyped-call]

    @staticmethod
    def sum_set(expression: sp.Expr, variables: list[str], _latex: str, callback: SetCallback) -> sp.Expr:
        # TODO use latex output
        variable_symbols = [_FormulaHelpers.variable(v) for v in variables]
        assignments = [x if isinstance(x, tuple) else (x,) for x in callback()]
        return cast(
            sp.Expr,
            functools.reduce(
                lambda total, new: total + expression.subs(dict(zip(variable_symbols, new))),  # type: ignore[no-untyped-call]
                assignments,
                sp.Integer(0),
            ),
        )

    @staticmethod
    def adjacency(v: int | str | sp.Expr, w: int | str | sp.Expr) -> sp.Function:
        if isinstance(v, str):
            v = _FormulaHelpers.variable(v)
        if isinstance(w, str):
            w = _FormulaHelpers.variable(w)
        return cast(sp.Function, A(v, w))

    @staticmethod
    def variable(name: str) -> sp.Symbol:
        return sp.Symbol(name)  # type: ignore[no-untyped-call]

    @staticmethod
    def get_encoding_variable_one_hot(path: Any, vertex: Any, position: Any) -> sp.Function:
        if isinstance(path, str):
            path = _FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = _FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = _FormulaHelpers.variable(position)
        return cast(sp.Function, X(path, vertex, position))

    @staticmethod
    def get_encoding_variable_unary(path: Any, vertex: Any, position: Any) -> sp.Expr:
        if isinstance(path, str):
            path = _FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = _FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = _FormulaHelpers.variable(position)
        return cast(
            sp.Expr,
            _FormulaHelpers.get_encoding_variable_one_hot(path, vertex, position)
            - _FormulaHelpers.get_encoding_variable_one_hot(path, vertex + 1, position),
        )

    @staticmethod
    def get_pos_is_not_unary(path: Any, position: Any, vertex: Any, num_vertices: int) -> sp.Expr:
        if isinstance(path, str):
            path = _FormulaHelpers.variable(path)
        if isinstance(position, str):
            position = _FormulaHelpers.variable(position)
        if isinstance(vertex, str):
            vertex = _FormulaHelpers.variable(vertex)
        if vertex == num_vertices:
            return cast(sp.Expr, 1 - _FormulaHelpers.get_encoding_variable_one_hot(path, vertex, position))
        return cast(
            sp.Expr,
            _FormulaHelpers.get_encoding_variable_one_hot(path, vertex, position)
            * (1 - _FormulaHelpers.get_encoding_variable_one_hot(path, vertex + 1, position)),
        )

    @staticmethod
    def get_for_each_path(expression: sp.Expr, paths: list[int]) -> sp.Expr:
        return _FormulaHelpers.sum_set(
            expression,
            ["p"],
            rf"\in \left\{{{','.join([str(p) for p in paths])}\right\}}",
            lambda: list(paths),
        )

    @staticmethod
    def get_for_each_position(expression: sp.Expr, path_size: int) -> sp.Expr:
        return _FormulaHelpers.sum_from_to(expression, "i", 1, path_size)

    @staticmethod
    def get_for_each_vertex(expression: sp.Expr, vertices: list[int]) -> sp.Expr:
        return _FormulaHelpers.sum_set(
            expression,
            ["v"],
            rf"\in \left\{{{','.join([str(v) for v in vertices])}\right\}}",
            lambda: list(vertices),
        )


class CostFunction(ABC):
    def get_formula(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        if settings.encoding_type == EncodingType.ONE_HOT:
            return self.get_formula_one_hot(graph, settings)
        if settings.encoding_type == EncodingType.UNARY:
            return self.get_formula_unary(graph, settings)
        msg = f"Encoding type {settings.encoding_type} not supported."
        raise ValueError(msg)

    @abstractmethod
    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        pass

    def get_formula_one_hot(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(
            graph, settings, lambda p, v, i: _FormulaHelpers.get_encoding_variable_one_hot(p, v, i)
        )

    def get_formula_unary(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(
            graph, settings, lambda p, v, i: _FormulaHelpers.get_encoding_variable_unary(p, v, i)
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class CompositeCostFunction(CostFunction):
    summands: list[tuple[CostFunction, int]]

    def __init__(self, *parts: tuple[CostFunction, int]) -> None:
        self.summands = list(parts)

    def __str__(self) -> str:
        return "   " + "\n + ".join([f"{w} * {fn}" for (fn, w) in self.summands])

    def get_formula(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return cast(
            sp.Expr,
            functools.reduce(
                lambda a, b: a + b[1] * b[0].get_formula(graph, settings),
                self.summands[1:],
                self.summands[0][1] * self.summands[0][0].get_formula(graph, settings),
            ),
        )

    def get_formula_general(
        self, _graph: Graph, _settings: PathFindingQUBOGeneratorSettings, _get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        msg = "This method should not be called for a composite cost function."
        raise RuntimeError(msg)


class PathPositionIs(CostFunction):
    vertex_ids: list[int]
    path: int
    position: int

    def __init__(self, position: int, vertex_ids: list[int], path: int) -> None:
        self.vertex_ids = vertex_ids
        self.position = position
        self.path = path

    def get_formula_general(
        self, _graph: Graph, _settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        return cast(
            sp.Expr,
            (
                1
                - _FormulaHelpers.sum_set(
                    get_variable_function(self.path, "v", self.position),
                    ["v"],
                    f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                    lambda: list(self.vertex_ids),
                )
            )
            ** 2,
        )

    def __str__(self) -> str:
        return f"PathPosition[{self.position}]Is[{','.join([str(v) for v in self.vertex_ids])}]"


class PathStartsAt(PathPositionIs):
    vertex_ids: list[int]

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        super().__init__(1, vertex_ids, path)

    def __str__(self) -> str:
        return f"PathStartsAt[{','.join([str(v) for v in self.vertex_ids])}]"


class PathEndsAt(PathPositionIs):
    vertex_ids: list[int]

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        super().__init__(-1, vertex_ids, path)

    def __str__(self) -> str:
        return f"PathEndsAt[{','.join([str(v) for v in self.vertex_ids])}]"


class PathContainsVertices(CostFunction):
    vertex_ids: list[int]
    min_occurrences: int
    max_occurrences: int
    possible_paths: list[int]

    def __init__(
        self,
        min_occurrences: int,
        max_occurrences: int,
        vertex_ids: list[int],
        possible_paths: list[int],
    ) -> None:
        self.vertex_ids = vertex_ids
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.possible_paths = possible_paths

    def __str__(self) -> str:
        vertices = ",".join([str(v) for v in self.vertex_ids])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"


class PathContainsVerticesExactlyOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], possible_paths: list[int]) -> None:
        super().__init__(1, 1, vertex_ids, possible_paths)

    def get_formula_general(
        self, _graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            (
                _FormulaHelpers.sum_set(
                    (
                        1
                        - _FormulaHelpers.get_for_each_position(
                            get_variable_function("p", "v", "i"), settings.max_path_length
                        )
                    )
                    ** 2,
                    ["v"],
                    f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                    lambda: list(self.vertex_ids),
                )
            ),
            self.possible_paths,
        )


class PathContainsVerticesAtLeastOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], possible_paths: list[int]) -> None:
        super().__init__(1, -1, vertex_ids, possible_paths)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathContainsVerticesAtMostOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], possible_paths: list[int]) -> None:
        super().__init__(0, 1, vertex_ids, possible_paths)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathContainsEdges(CostFunction):
    edges: list[tuple[int, int]]
    min_occurrences: int
    max_occurrences: int
    possible_paths: list[int]

    def __init__(
        self,
        min_occurrences: int,
        max_occurrences: int,
        edges: list[tuple[int, int]],
        possible_paths: list[int],
    ) -> None:
        self.edges = edges
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.possible_paths = possible_paths

    def __str__(self) -> str:
        vertices = ",".join([str(v) for v in self.edges])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"


class PathContainsEdgesExactlyOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], possible_paths: list[int]) -> None:
        super().__init__(1, 1, edges, possible_paths)

    def get_formula_general(
        self, graph: Graph, _settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                (
                    1
                    - _FormulaHelpers.sum_from_to(
                        get_variable_function("p", "v", "i")
                        * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1),
                        "i",
                        1,
                        graph.n_vertices - 1,  # TODO use path length instead
                    )
                )
                ** 2,
                ["v", "w"],
                f"\\in \\left\\{{ {', '.join(['(' + str(v) + ', ' + str(w) + ')' for (v, w) in self.edges])} \\right\\}}",
                lambda: list(self.edges),
            ),
            self.possible_paths,
        )


class PathContainsEdgesAtLeastOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], possible_paths: list[int]) -> None:
        super().__init__(1, -1, edges, possible_paths)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathContainsEdgesAtMostOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], possible_paths: list[int]) -> None:
        super().__init__(0, 1, edges, possible_paths)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathBound(CostFunction):
    path_ids: list[int]

    def __init__(self, path_ids: list[int]) -> None:
        self.path_ids = path_ids

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{','.join([str(path_id) for path_id in self.path_ids])}]"


class PrecedenceConstraint(PathBound):
    pre: int
    post: int

    def __init__(self, pre: int, post: int, path_ids: list[int]) -> None:
        super().__init__(path_ids)
        self.pre = pre
        self.post = post

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathsShareNoVertices(PathBound):
    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathsShareNoEdges(PathBound):
    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        raise NotImplementedError


class PathIsValid(PathBound):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                _FormulaHelpers.get_for_each_position(
                    get_variable_function("p", "v", "i")
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1),
                    settings.max_path_length,
                ),
                ["v", "w"],
                "\\not\\in E",
                lambda: [
                    (i + 1, j + 1)
                    for i in range(graph.n_vertices)
                    for j in range(graph.n_vertices)
                    if graph.adjacency_matrix[i, j] <= 0
                ],
            ),
            self.path_ids,
        )

    def get_formula_one_hot(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        def get_variable_function(p: Any, v: Any, i: Any) -> sp.Expr:
            return _FormulaHelpers.get_encoding_variable_one_hot(p, v, i)

        general = self.get_formula_general(graph, settings, get_variable_function)
        return cast(
            sp.Expr,
            general
            + _FormulaHelpers.get_for_each_path(
                _FormulaHelpers.get_for_each_position(
                    (1 - _FormulaHelpers.get_for_each_vertex(get_variable_function("p", "v", "i"), graph.all_vertices))
                    ** 2,
                    settings.max_path_length,
                ),
                self.path_ids,
            ),
        )

    def get_formula_unary(self, graph: Graph, settings: PathFindingQUBOGeneratorSettings) -> sp.Expr:
        general = self.get_formula_general(
            graph, settings, lambda p, v, i: _FormulaHelpers.get_encoding_variable_unary(p, v, i)
        )
        return cast(
            sp.Expr,
            general
            + _FormulaHelpers.get_for_each_path(
                _FormulaHelpers.get_for_each_position(
                    1 - _FormulaHelpers.get_encoding_variable_one_hot("p", 1, "i"), settings.max_path_length
                ),
                self.path_ids,
            )
            + 2
            * _FormulaHelpers.get_for_each_path(
                _FormulaHelpers.get_for_each_position(
                    _FormulaHelpers.sum_set(
                        (1 - _FormulaHelpers.get_encoding_variable_one_hot("p", "v", "i"))
                        * _FormulaHelpers.get_encoding_variable_one_hot("p", _FormulaHelpers.variable("v") + 1, "i"),
                        ["v"],
                        "\\in V",
                        cast(SetCallback, lambda: graph.all_vertices),
                    ),
                    settings.max_path_length,
                ),
                self.path_ids,
            ),  # TODO prove this is enough
        )


class MinimisePathLength(PathBound):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)

    def get_formula_general(
        self, graph: Graph, settings: PathFindingQUBOGeneratorSettings, get_variable_function: GetVariableFunction
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                _FormulaHelpers.get_for_each_position(
                    _FormulaHelpers.adjacency("v", "w")
                    * get_variable_function("p", "v", "i")
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1),
                    settings.max_path_length,
                ),
                ["v", "w"],
                "\\in E",
                lambda: [
                    (i + 1, j + 1)
                    for i in range(graph.n_vertices)
                    for j in range(graph.n_vertices)
                    if graph.adjacency_matrix[i, j] > 0
                ],
            ),
            self.path_ids,
        )


def merge(cost_functions: list[CostFunction], optimisation_goals: list[CostFunction]) -> CompositeCostFunction:
    return CompositeCostFunction(*([(f, 1) for f in cost_functions] + [(f, 1) for f in optimisation_goals]))


@dataclass
class PathFindingQUBOGeneratorSettings:
    encoding_type: EncodingType
    n_paths: int
    max_path_length: int
    loops: bool = False


class PathFindingQUBOGenerator(QUBOGenerator):
    graph: Graph
    settings: PathFindingQUBOGeneratorSettings

    def __init__(
        self,
        objective_function: CostFunction | None,
        graph: Graph,
        settings: PathFindingQUBOGeneratorSettings,
    ) -> None:
        super().__init__(objective_function.get_formula(graph, settings) if objective_function is not None else None)
        self.graph = graph
        self.settings = settings

    def add_constraint(self, constraint: CostFunction) -> Self:
        self.add_penalty(constraint.get_formula(self.graph, self.settings))
        return self

    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        return [(expr, lam if lam else self.__optimal_lambda()) for (expr, lam) in self.penalties]

    def __optimal_lambda(self) -> float:
        return cast(float, np.sum(self.graph.adjacency_matrix))

    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        assignment = [
            (_FormulaHelpers.adjacency(i + 1, j + 1), self.graph.adjacency_matrix[i, j])
            for i in range(self.graph.n_vertices)
            for j in range(self.graph.n_vertices)
        ]
        assignment += [
            (_FormulaHelpers.get_encoding_variable_one_hot(p + 1, self.graph.n_vertices + 1, i + 1), 0)
            for p in range(self.settings.n_paths)
            for i in range(self.settings.max_path_length + 1)
        ]  # x_{p, |V| + 1, i} = 0 for all p, i
        assignment += [
            (
                _FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, self.settings.max_path_length + 1),
                _FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, 1) if self.settings.loops else 0,
            )
            for p in range(self.settings.n_paths)
            for v in range(self.graph.n_vertices)
        ]  # x_{p, v, N + 1} = x_{p, v, 1} for all p, v if loop, otherwise 0
        result = expression.subs(dict(assignment))  # type: ignore[no-untyped-call]
        if isinstance(result, sp.Expr):
            return result
        msg = "Expression is not an expression."
        raise ValueError(msg)

    def get_variable_index(self, var: sp.Function) -> int:
        parts = var.args

        if any(not isinstance(part, sp.core.Integer) for part in parts):
            msg = "Variable subscripts must be integers."
            raise ValueError(msg)

        p = int(cast(int, parts[0]))
        v = int(cast(int, parts[1]))
        i = int(cast(int, parts[2]))

        return int(
            (v - 1)
            + (i - 1) * self.settings.max_path_length
            + (p - 1) * self.settings.max_path_length * self.graph.n_vertices
            + 1
        )

    def decode_bit_array(self, _array: list[int]) -> Any:
        if self.settings.encoding_type == EncodingType.ONE_HOT:
            return self.decode_bit_array_one_hot(_array)
        if self.settings.encoding_type == EncodingType.UNARY:
            return self.decode_bit_array_unary(_array)
        msg = f"Encoding type {self.settings.encoding_type} not supported."
        raise ValueError(msg)

    def decode_bit_array_unary(self, array: list[int]) -> Any:
        paths = []
        for p in range(self.settings.n_paths):
            path = []
            for i in range(self.settings.max_path_length):
                c = 0
                for v in range(self.graph.n_vertices):
                    c += array[
                        v + i * self.graph.n_vertices + p * self.graph.n_vertices * self.settings.max_path_length
                    ]
                path.append(c)
            paths.append(path)
        return paths

    def decode_bit_array_one_hot(self, array: list[int]) -> Any:
        path = []
        for i, bit in enumerate(array):
            if bit == 0:
                continue
            v = i % self.graph.n_vertices
            s = i // self.graph.n_vertices
            path.append((v, s))
        path.sort(key=lambda x: x[1])
        return [entry[0] + 1 for entry in path]

    def _get_all_variables(self) -> Sequence[tuple[sp.Expr, int]]:
        result = []
        for p in range(self.settings.n_paths):
            for v in self.graph.all_vertices:
                for i in range(self.settings.max_path_length):
                    var = _FormulaHelpers.get_encoding_variable_one_hot(p + 1, v, i + 1)
                    result.append((var, self.get_variable_index(var)))
        return result
