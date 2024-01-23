from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from mqt.qubomaker import Graph
    from mqt.qubomaker.pathfinder import pathfinder

SetCallback = Callable[[], list[sp.Expr | int | float | tuple[sp.Expr | int | float, ...]]]
GetVariableFunction = Callable[[Any, Any, Any, int], sp.Expr]


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


class Decompose(sp.Function):
    def _latex(self, _printer: sp.StrPrinter, *_args: Any, **_kwargs: Any) -> str:
        n, i = (self.args[0], self.args[1])
        return rf"\bar{{{n}}}_{{{i}}}"

    def doit(self, **_hints: Any) -> sp.Expr:
        n, i = self.args
        if not isinstance(n, sp.Integer) or not isinstance(i, sp.Integer):
            return self
        return cast(sp.Expr, sp.Integer(int(n) >> (int(i) - 1) & 1))


class _FormulaHelpers:
    @staticmethod
    def sum_from_to(expression: sp.Expr, var: str, from_number: int, to_number: int) -> sp.Expr:
        s = sp.Symbol(var)  # type: ignore[no-untyped-call]
        return sp.Sum(expression, (s, from_number, to_number))  # type: ignore[no-untyped-call]

    @staticmethod
    def prod_from_to(expression: sp.Expr, var: str, from_number: int, to_number: int) -> sp.Expr:
        s = sp.Symbol(var)  # type: ignore[no-untyped-call]
        return sp.Product(expression, (s, from_number, to_number))  # type: ignore[no-untyped-call]

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
    def get_encoding_variable_one_hot(path: Any, vertex: Any, position: Any, _num_vertices: int = 0) -> sp.Function:
        if isinstance(path, str):
            path = _FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = _FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = _FormulaHelpers.variable(position)
        return cast(sp.Function, X(path, vertex, position))

    @staticmethod
    def get_encoding_variable_unary(path: Any, vertex: Any, position: Any, _num_vertices: int = 0) -> sp.Expr:
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
    def get_encoding_variable_binary(path: Any, vertex: Any, position: Any, num_vertices: int = 0) -> sp.Expr:
        if isinstance(path, str):
            path = _FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = _FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = _FormulaHelpers.variable(position)
        index_symbol = _FormulaHelpers.variable("v")
        if index_symbol == vertex:
            index_symbol = _FormulaHelpers.variable("w")
        max_index = int(np.ceil(np.log2(num_vertices)))
        return cast(
            sp.Expr,
            sp.Product(
                Decompose(vertex - 1, index_symbol)
                * _FormulaHelpers.get_encoding_variable_one_hot(path, index_symbol, position)
                + (1 - Decompose(vertex - 1, index_symbol))
                * (1 - _FormulaHelpers.get_encoding_variable_one_hot(path, index_symbol, position)),
                (index_symbol, 1, max_index),
            ),  # type: ignore[no-untyped-call]
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
    def get_formula(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        if settings.encoding_type == EncodingType.ONE_HOT:
            return self.get_formula_one_hot(graph, settings)
        if settings.encoding_type == EncodingType.UNARY:
            return self.get_formula_unary(graph, settings)
        if settings.encoding_type == EncodingType.BINARY:
            return self.get_formula_binary(graph, settings)
        return None  # type: ignore[unreachable]

    @abstractmethod
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        pass

    def get_formula_one_hot(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(graph, settings, _FormulaHelpers.get_encoding_variable_one_hot)

    def get_formula_unary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(graph, settings, _FormulaHelpers.get_encoding_variable_unary)

    def get_formula_binary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(graph, settings, _FormulaHelpers.get_encoding_variable_binary)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class CompositeCostFunction(CostFunction):
    summands: list[tuple[CostFunction, int]]

    def __init__(self, *parts: tuple[CostFunction, int]) -> None:
        self.summands = list(parts)

    def __str__(self) -> str:
        return "   " + "\n + ".join([f"{w} * {fn}" for (fn, w) in self.summands])

    def get_formula(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return cast(
            sp.Expr,
            functools.reduce(
                lambda a, b: a + b[1] * b[0].get_formula(graph, settings),
                self.summands[1:],
                self.summands[0][1] * self.summands[0][0].get_formula(graph, settings),
            ),
        )

    def get_formula_general(
        self,
        _graph: Graph,
        _settings: pathfinder.PathFindingQUBOGeneratorSettings,
        _get_variable_function: GetVariableFunction,
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
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return cast(
            sp.Expr,
            (
                1
                - _FormulaHelpers.sum_set(
                    get_variable_function(
                        self.path,
                        "v",
                        self.position if self.position > 0 else (settings.max_path_length + 1 + self.position),
                        graph.n_vertices,
                    ),
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
    path_ids: list[int]

    def __init__(
        self,
        min_occurrences: int,
        max_occurrences: int,
        vertex_ids: list[int],
        path_ids: list[int],
    ) -> None:
        self.vertex_ids = vertex_ids
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.path_ids = path_ids

    def __str__(self) -> str:
        vertices = ",".join([str(v) for v in self.vertex_ids])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"

    def _handle_for_each(self, expression: sp.Expr) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            (
                _FormulaHelpers.sum_set(
                    expression,
                    ["v"],
                    f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                    lambda: list(self.vertex_ids),
                )
            ),
            self.path_ids,
        )


class PathContainsVerticesExactlyOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        super().__init__(1, 1, vertex_ids, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            (
                1
                - _FormulaHelpers.get_for_each_position(
                    get_variable_function("p", "v", "i", graph.n_vertices), settings.max_path_length
                )
            )
            ** 2
        )


class PathContainsVerticesAtLeastOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        super().__init__(1, -1, vertex_ids, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            _FormulaHelpers.prod_from_to(
                (1 - get_variable_function("p", "v", "i", graph.n_vertices)), "i", 1, settings.max_path_length
            )
        )


class PathContainsVerticesAtMostOnce(PathContainsVertices):
    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        super().__init__(0, 1, vertex_ids, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            _FormulaHelpers.sum_from_to(
                _FormulaHelpers.sum_from_to(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "v", "j", graph.n_vertices),
                    "j",
                    _FormulaHelpers.variable("i") + 1,
                    settings.max_path_length,
                ),
                "i",
                1,
                settings.max_path_length - 1,
            )
        )


class PathContainsEdges(CostFunction):
    edges: list[tuple[int, int]]
    min_occurrences: int
    max_occurrences: int
    path_ids: list[int]

    def __init__(
        self,
        min_occurrences: int,
        max_occurrences: int,
        edges: list[tuple[int, int]],
        path_ids: list[int],
    ) -> None:
        self.edges = edges
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.path_ids = path_ids

    def __str__(self) -> str:
        vertices = ",".join([str(v) for v in self.edges])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"

    def _handle_for_each(self, expression: sp.Expr) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                expression,
                ["v", "w"],
                f"\\in \\left\\{{ {', '.join(['(' + str(v) + ', ' + str(w) + ')' for (v, w) in self.edges])} \\right\\}}",
                lambda: list(self.edges),
            ),
            self.path_ids,
        )


class PathContainsEdgesExactlyOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        super().__init__(1, 1, edges, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            (
                1
                - _FormulaHelpers.sum_from_to(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    "i",
                    1,
                    settings.max_path_length,
                )
            )
            ** 2
        )


class PathContainsEdgesAtLeastOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        super().__init__(1, -1, edges, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            _FormulaHelpers.prod_from_to(
                (
                    1
                    - get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            )
        )


class PathContainsEdgesAtMostOnce(PathContainsEdges):
    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        super().__init__(0, 1, edges, path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            _FormulaHelpers.sum_from_to(
                _FormulaHelpers.sum_from_to(
                    (
                        get_variable_function("p", "v", "i", graph.n_vertices)
                        * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices)
                        * get_variable_function("p", "v", "j", graph.n_vertices)
                        * get_variable_function("p", "w", _FormulaHelpers.variable("j") + 1, graph.n_vertices)
                    ),
                    "j",
                    _FormulaHelpers.variable("i") + 1,
                    settings.max_path_length,
                ),
                "i",
                1,
                settings.max_path_length - 1,
            )
        )


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
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_from_to(
                get_variable_function("p", self.post, "i", graph.n_vertices)
                * _FormulaHelpers.prod_from_to(
                    (1 - get_variable_function("p", self.pre, "j", graph.n_vertices)),
                    "j",
                    1,
                    _FormulaHelpers.variable("i") - 1,
                ),
                "i",
                1,
                settings.max_path_length,
            ),
            self.path_ids,
        )


class PathComparison(CostFunction):
    path_one: int
    path_two: int

    def __init__(self, path_one: int, path_two: int) -> None:
        self.path_one = path_one
        self.path_two = path_two

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.path_one}, {self.path_two}]"


class PathsShareNoVertices(PathComparison):
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_vertex(
            _FormulaHelpers.get_for_each_position(
                get_variable_function(self.path_one, "v", "i", graph.n_vertices), settings.max_path_length
            )
            * _FormulaHelpers.get_for_each_position(
                get_variable_function(self.path_two, "v", "i", graph.n_vertices), settings.max_path_length
            ),
            graph.all_vertices,
        )


class PathsShareNoEdges(PathComparison):
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return _FormulaHelpers.sum_set(
            _FormulaHelpers.sum_from_to(
                (
                    get_variable_function(self.path_one, "v", "i", graph.n_vertices)
                    * get_variable_function(self.path_one, "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            )
            * _FormulaHelpers.sum_from_to(
                (
                    get_variable_function(self.path_two, "v", "i", graph.n_vertices)
                    * get_variable_function(self.path_two, "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            ),
            ["v", "w"],
            "\\in E",
            lambda: cast(list[sp.Expr | int | float | tuple[sp.Expr | int | float, ...]], graph.all_edges),
        )


class PathIsValid(PathBound):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                _FormulaHelpers.get_for_each_position(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices),
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

    def get_formula_one_hot(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        def get_variable_function(p: Any, v: Any, i: Any, _n: int = 0) -> sp.Expr:
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

    def get_formula_unary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        general = self.get_formula_general(graph, settings, _FormulaHelpers.get_encoding_variable_unary)
        enforce_domain_wall_penalty = (
            2 * settings.max_path_length * np.max(graph.adjacency_matrix) + graph.n_vertices**2
        )
        # This ensures that the domain wall condition (x_i = 0 -> x_{i+1} = 0) is not broken to achieve better cost in other cost functions.
        return cast(
            sp.Expr,
            general
            + _FormulaHelpers.get_for_each_path(
                _FormulaHelpers.get_for_each_position(
                    1 - _FormulaHelpers.get_encoding_variable_one_hot("p", 1, "i"), settings.max_path_length
                ),
                self.path_ids,
            )  # Enforce \pi_1 >= 1
            + enforce_domain_wall_penalty
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

    def get_formula_binary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(graph, settings, _FormulaHelpers.get_encoding_variable_binary)


class MinimisePathLength(PathBound):
    def __init__(self, path_ids: list[int]) -> None:
        super().__init__(path_ids)

    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return _FormulaHelpers.get_for_each_path(
            _FormulaHelpers.sum_set(
                _FormulaHelpers.get_for_each_position(
                    _FormulaHelpers.adjacency("v", "w")
                    * get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", _FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    settings.max_path_length,
                ),
                ["v", "w"],
                "\\in E",
                lambda: cast(list[sp.Expr | int | float | tuple[sp.Expr | int | float, ...]], graph.all_edges),
            ),
            self.path_ids,
        )


def merge(cost_functions: list[CostFunction], optimisation_goals: list[CostFunction]) -> CompositeCostFunction:
    return CompositeCostFunction(*([(f, 1) for f in cost_functions] + [(f, 1) for f in optimisation_goals]))
