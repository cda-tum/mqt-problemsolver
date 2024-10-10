"""Includes all cost functions supported by the pathfinding QUBOGenerator."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np
import sympy as sp
from typing_extensions import override

if TYPE_CHECKING:
    from mqt.qubomaker import Graph
    from mqt.qubomaker.pathfinder import pathfinder

    GetVariableFunction = Callable[[Any, Any, Any, int], sp.Expr]

SetCallback = Callable[[], list[Union[sp.Expr, int, float, tuple[Union[sp.Expr, int, float], ...]]]]


class EncodingType(Enum):
    """The encoding types supported for the QUBO generation."""

    ONE_HOT = 1
    DOMAIN_WALL = 2
    BINARY = 3


class A(sp.Function):
    """Custom sympy function for the adjacency matrix.

    `A(v, w)` represents the weight of edge (v, w).
    """

    def _latex(self, printer: sp.StrPrinter, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        """Returns the latex representation of the expression.

        Args:
            printer (sp.StrPrinter): The printer to use.
            args (Any): Additional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: The latex representation of the expression.
        """
        v, w = (self.args[0], self.args[1])
        return rf"A_{{{v},{w}}}"


class X(sp.Function):
    """Custom sympy function for the encoding variables.

    `x(p, v, i)` represents a single binary variable. Its meaning depends on the encoding type.
    """

    def _latex(self, printer: sp.StrPrinter, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        """Returns the latex representation of the expression.

        Args:
            printer (sp.StrPrinter): The printer to use.
            args (Any): Additional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: The latex representation of the expression.
        """
        p, v, i = (self.args[0], self.args[1], self.args[2])
        return rf"x_{{{p},{v},{i}}}"


class Decompose(sp.Function):
    """Custom sympy function for the decomposition of an integer into its binary digits.

    Example:
        `Decompose(5, 1)` represents digit 1 of the binary string `101`, i.e., 1.
    """

    def _latex(self, printer: sp.StrPrinter, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        """Returns the latex representation of the expression.

        Args:
            printer (sp.StrPrinter): The printer to use.
            args (Any): Additional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: The latex representation of the expression.
        """
        n, i = (self.args[0], self.args[1])
        return rf"\bar{{{n}}}_{{{i}}}"

    @override
    def doit(self, **_hints: Any) -> sp.Expr:
        n, i = self.args
        if not isinstance(n, sp.Integer) or not isinstance(i, sp.Integer):
            return self
        return cast(sp.Expr, sp.Integer(int(n) >> (int(i) - 1) & 1))


class ExpandingSum(sp.Sum):
    """Represents a sum that is always expanded into a sum of its elements when calling `doit()`."""

    args: tuple[sp.Expr, tuple[sp.Symbol, sp.Expr, sp.Expr]]

    @override
    def doit(self, **_hints: Any) -> sp.Expr:
        expr, *limits = self.args
        limits = list(reversed(limits))
        return self.__do_expansion(expr, limits[1:], limits[0])

    def __do_expansion(
        self,
        expr: sp.Expr | sp.Basic,
        remaining_limits: list[tuple[sp.Symbol, sp.Expr, sp.Expr]],
        limits: tuple[sp.Symbol, sp.Expr, sp.Expr],
    ) -> sp.Expr:
        """Expands one layer of the sum.

        A sum may consist of multiple layers of sums. This method expands the outer-most layer of the sum. And then continues
        recursively with the remaining layers.

        Args:
            expr (sp.Expr | sp.Basic): The expression inside the sum.
            remaining_limits (list[tuple[sp.Symbol, sp.Expr, sp.Expr]]): The remaining sum limits that have not been expanded yet.
            limits (tuple[sp.Symbol, sp.Expr, sp.Expr]): The sum limits that are expanded in this step.

        Returns:
            sp.Expr: The expanded sum of elements.
        """
        result: sp.Expr = sp.Integer(0)
        (variable, from_value, to_value) = limits
        if not isinstance(from_value, sp.Integer) or not isinstance(to_value, sp.Integer):
            return ExpandingSum(expr, *list(reversed([limits, *remaining_limits])))
        for i in range(int(from_value), int(to_value) + 1):
            if len(remaining_limits) == 0:
                result += expr.subs(variable, i)
            else:
                new_remaining_limits = [
                    (new_variable, new_from_limit.subs(variable, i), new_to_limit.subs(variable, i))
                    for (new_variable, new_from_limit, new_to_limit) in remaining_limits
                ]
                result += self.__do_expansion(expr.subs(variable, i), new_remaining_limits[1:], new_remaining_limits[0])
        return result


class _StringForSumSet:
    """A string that can be stored in a SumSet object.

    Required, as storing just a normal `str` is not compatible with sympy.
    """

    string: str
    args: list[sp.Expr]
    free_symbols: list[sp.Symbol]

    def __init__(self, string: str) -> None:
        """Initialises a _StringForSumSet.

        Args:
            string (str): The string to store.
        """
        self.string = string
        self.args = []
        self.free_symbols = []

    def __str__(self) -> str:
        """Returns a string representation of the expression.

        Returns:
            str: The string represented by this expression.
        """
        return self.string


class SumSet(sp.Expr):
    """A class  that can be used to represent a sum over a set.

    This is just a symbolic representation for the display of the equation. In the background,
    it stores the actual expanded sum that is used for calculations.
    """

    expr: sp.Expr
    element_expr: sp.Expr
    latex: _StringForSumSet

    def __init__(self, expr: sp.Expr, element_expr: sp.Expr, latex: _StringForSumSet) -> None:
        """Initialises a SumSet.

        Args:
            expr (sp.Expr): The expression that is represented by the SumSet.
            element_expr (sp.Expr): The expression of an individual sum item.
            latex (_StringForSumSet): The latex string that represents the set over which the sum is performed.
        """
        self._args = (expr, element_expr, latex)
        self.expr = expr
        self.latex = latex
        self.element_expr = element_expr

    def _latex(self, printer: sp.StrPrinter, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
        """Returns the latex representation of the expression.

        Args:
            printer (sp.StrPrinter): The printer to use.
            args (Any): Additional arguments.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: The latex representation of the expression.
        """
        child_latex = printer.doprint(self.element_expr)
        return f"{self.latex.string} {child_latex}"

    @override
    def doit(self, **hints: Any) -> sp.Expr:
        """Replaces the sum by the actual expression it represents.

        Returns:
            sp.Expr: The expression that is represented by the sum.
        """
        return self.expr.doit(hints=hints)

    def __eq__(self, other: object) -> bool:
        """Overrides the default implementation."""
        if not isinstance(other, SumSet):
            return False
        return self.expr == other.expr

    def __hash__(self) -> int:
        """Overrides the default implementation."""
        return hash(self.expr)


class FormulaHelpers:
    """Provides static methods for the more efficient construction of sympy formulas."""

    @staticmethod
    def sum_from_to(expression: sp.Expr, var: str, from_number: sp.Expr | float, to_number: sp.Expr | float) -> sp.Expr:
        """Generates a sum of the form `Sum(expression, (var, from_number, to_number)`.

        Args:
            expression (sp.Expr): The term inside the sum.
            var (str): The iteration variable of the sum as a string.
            from_number (sp.Expr | float): The lower bound of the sum.
            to_number (sp.Expr | float): The upper bound of the sum.

        Returns:
            sp.Expr: The sympy sum term.
        """
        s = sp.Symbol(var)
        return ExpandingSum(expression, (s, from_number, to_number))

    @staticmethod
    def prod_from_to(
        expression: sp.Expr, var: str, from_number: sp.Expr | float, to_number: sp.Expr | float
    ) -> sp.Expr:
        """Generates a product of the form `Product(expression, (var, from_number, to_number)`.

        Args:
            expression (sp.Expr): The term inside the sum.
            var (str): The iteration variable of the sum as a string.
            from_number (sp.Expr | float): The lower bound of the sum.
            to_number (sp.Expr | float): The upper bound of the sum.

        Returns:
            sp.Expr: The sympy sum term.
        """
        s = sp.Symbol(var)
        return sp.Product(expression, (s, from_number, to_number))

    @staticmethod
    def sum_set(expression: sp.Expr, variables: list[str], latex: str, callback: SetCallback) -> sp.Expr:
        r"""Generates a sum of the form `\sum_{[variables] \in [callback]} [expression]`.

        Args:
            expression (sp.Expr): The term inside the sum.
            variables (list[str]): A list of iteration variables of the sum as strings.
            latex (str): The latex representation of the set expression.
            callback (SetCallback): A callback returning the set over which the sum should iterate.

        Returns:
            sp.Expr: The sympy sum term.
        """
        variable_symbols = [FormulaHelpers.variable(v) for v in variables]
        assignments = [x if isinstance(x, tuple) else (x,) for x in callback()]
        expr = functools.reduce(
            lambda total, new: total + expression.subs(dict(zip(variable_symbols, new))),
            assignments,
            cast(sp.Expr, sp.Integer(0)),
        )

        if len(assignments) <= 1:
            return expr

        if len(variables) == 1:
            iterator_latex = str(variables[0])
        else:
            iterator_latex = "(" + ", ".join([str(v) for v in variable_symbols]) + ")"

        return SumSet(expr, expression, _StringForSumSet(rf"\sum_{{{iterator_latex} {latex}}}"))

    @staticmethod
    def adjacency(v: int | str | sp.Expr, w: int | str | sp.Expr) -> sp.Expr:
        """Returns an access to the adjacency matrix with indices `v` and `w`.

        `v` and `w` may be of type int, str, or sympy expressions.

        Args:
            v (int | str | sp.Expr): The "from" index of the adjacency matrix.
            w (int | str | sp.Expr): The "to" index of the adjacency matrix.

        Returns:
            sp.Expr: A call to the adjacency matrix function.
        """
        if isinstance(v, str):
            v = FormulaHelpers.variable(v)
        if isinstance(w, str):
            w = FormulaHelpers.variable(w)
        return cast(sp.Expr, A(v, w))

    @staticmethod
    def variable(name: str) -> sp.Symbol:
        """Returns a variable with the given name.

        Args:
            name (str): The name of the variable.

        Returns:
            sp.Symbol: The generated variable.
        """
        return sp.Symbol(name)

    @staticmethod
    def get_encoding_variable_one_hot(path: Any, vertex: Any, position: Any, _num_vertices: int = 0) -> sp.Expr:
        """Returns an access to the binary variable `x_{path, vertex, position}` that represents the statement "Vertex `vertex` is located at position `position` in path `path`" for One-Hot encoding.

        All indices can be integers, strings, or sympy expressions.
        The `_num_vertices` parameter is only included for compatibility.

        Args:
            path (Any): The path index.
            vertex (Any): The vertex index.
            position (Any): The position index.
            _num_vertices (int, optional): The number of vertices in the graph. Defaults to 0.

        Returns:
            sp.Expr: An expression representing the statement "Vertex `vertex` is located at position `position` in path `path`" for One-Hot encoding.
        """
        if isinstance(path, str):
            path = FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = FormulaHelpers.variable(position)
        return cast(sp.Expr, X(path, vertex, position))

    @staticmethod
    def get_encoding_variable_domain_wall(path: Any, vertex: Any, position: Any, _num_vertices: int = 0) -> sp.Expr:
        """Returns an expression representing the statement "Vertex `vertex` is located at position `position` in path `path`" for Domain-Wall encoding.

        All indices can be integers, strings, or sympy expressions.
        The `_num_vertices` parameter is only included for compatibility.

        Args:
            path (Any): The path index.
            vertex (Any): The vertex index.
            position (Any): The position index.
            _num_vertices (int, optional): The number of vertices in the graph. Defaults to 0.

        Returns:
            sp.Function: An expression representing the statement "Vertex `vertex` is located at position `position` in path `path`" for Domain-Wall encoding.
        """
        if isinstance(path, str):
            path = FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = FormulaHelpers.variable(position)
        return FormulaHelpers.get_encoding_variable_one_hot(
            path, vertex, position
        ) - FormulaHelpers.get_encoding_variable_one_hot(path, vertex + 1, position)

    @staticmethod
    def get_encoding_variable_binary(path: Any, vertex: Any, position: Any, num_vertices: int = 0) -> sp.Expr:
        """Returns an expression representing the statement "Vertex `vertex` is located at position `position` in path `path`" for Binary encoding.

        All indices can be integers, strings, or sympy expressions.
        The `_num_vertices` parameter is only included for compatibility.

        Args:
            path (Any): The path index.
            vertex (Any): The vertex index.
            position (Any): The position index.
            num_vertices (int, optional): The number of vertices in the graph. Defaults to 0.

        Returns:
            sp.Function: An expression representing the statement "Vertex `vertex` is located at position `position` in path `path`" for Binary encoding.
        """
        if isinstance(path, str):
            path = FormulaHelpers.variable(path)
        if isinstance(vertex, str):
            vertex = FormulaHelpers.variable(vertex)
        if isinstance(position, str):
            position = FormulaHelpers.variable(position)
        index_symbol = FormulaHelpers.variable("v")
        if index_symbol == vertex:
            index_symbol = FormulaHelpers.variable("w")
        max_index = int(np.ceil(np.log2(num_vertices + 1)))
        return cast(
            sp.Expr,
            sp.Product(
                Decompose(vertex, index_symbol)
                * FormulaHelpers.get_encoding_variable_one_hot(path, index_symbol, position)
                + (1 - Decompose(vertex, index_symbol))
                * (1 - FormulaHelpers.get_encoding_variable_one_hot(path, index_symbol, position)),
                (index_symbol, 1, max_index),
            ),
        )

    @staticmethod
    def get_for_each_path(expression: sp.Expr, paths: list[int]) -> sp.Expr:
        """Returns a sum iterating over the given expression for each path in `paths`.

        Args:
            expression (sp.Expr): The expression to sum up.
            paths (list[int]): The paths over which to sum the expression.

        Returns:
            sp.Expr: A sum of the expression for each path.
        """
        return FormulaHelpers.sum_set(
            expression,
            ["p"],
            rf"\in \left\{{{','.join([str(p) for p in paths])}\right\}}",
            lambda: list(paths),
        )

    @staticmethod
    def get_for_each_position(expression: sp.Expr, path_size: int) -> sp.Expr:
        """Returns a sum iterating over the given expression for each possible index in the path.

        Args:
            expression (sp.Expr): The expression to sum up.
            path_size (int): The maximum length of a path.

        Returns:
            sp.Expr: A sum of the expression for each position.
        """
        return FormulaHelpers.sum_from_to(expression, "i", 1, path_size)

    @staticmethod
    def get_for_each_vertex(expression: sp.Expr, vertices: list[int]) -> sp.Expr:
        """Returns a sum iterating over the given expression for each possible vertex in `vertices`.

        Args:
            expression (sp.Expr): The expression to sum up.
            vertices (list[int]): The list of vertices for which the expression should be summed.

        Returns:
            sp.Expr: A sum of the expression for each vertex.
        """
        return FormulaHelpers.sum_set(
            expression,
            ["v"],
            rf"\in \left\{{{','.join([str(v) for v in vertices])}\right\}}",
            lambda: list(vertices),
        )


class CostFunction(ABC):
    """An abstract base class for cost functions.

    Represents a cost function that can be translated into a QUBO expression.
    """

    def get_formula(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        """Translates the cost function into a QUBO expression.

        Args:
            graph (Graph): The graph on which the cost function should be applied.
            settings (pathfinder.PathFindingQUBOGeneratorSettings): The settings of the QUBO generator.

        Returns:
            sp.Expr: An expression representing the cost function as a QUBO function.
        """
        if settings.encoding_type == EncodingType.ONE_HOT:
            return self.get_formula_one_hot(graph, settings)
        if settings.encoding_type == EncodingType.DOMAIN_WALL:
            return self.get_formula_domain_wall(graph, settings)
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
        """Returns the QUBO expression for the cost function in the general case.

        In the general case, the cost function may access the blackbox function `get_variable_function(p, v, i)`
        that returns 1 if vertex `v` is located at position `i` in path `p` and 0 otherwise.

        Args:
            graph (Graph): The graph on which the cost function should be applied.
            settings (pathfinder.PathFindingQUBOGeneratorSettings): The settings of the QUBO generator.
            get_variable_function (GetVariableFunction): The blackbox function for accessing the encoding variables.

        Returns:
            sp.Expr: An expression representing the cost function as a QUBO function.
        """

    def get_formula_one_hot(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        """Computes the QUBO expression for the cost function for One-Hot encoding.

        Args:
            graph (Graph): The graph on which the cost function should be applied.
            settings (pathfinder.PathFindingQUBOGeneratorSettings): The settings of the QUBO generator.

        Returns:
            sp.Expr: An expression representing the cost function as a QUBO function.
        """
        return self.get_formula_general(graph, settings, FormulaHelpers.get_encoding_variable_one_hot)

    def get_formula_domain_wall(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        """Computes the QUBO expression for the cost function for Domain-Wall encoding.

        Args:
            graph (Graph): The graph on which the cost function should be applied.
            settings (pathfinder.PathFindingQUBOGeneratorSettings): The settings of the QUBO generator.

        Returns:
            sp.Expr: An expression representing the cost function as a QUBO function.
        """
        return self.get_formula_general(graph, settings, FormulaHelpers.get_encoding_variable_domain_wall)

    def get_formula_binary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        """Computes the QUBO expression for the cost function for Binary encoding.

        Args:
            graph (Graph): The graph on which the cost function should be applied.
            settings (pathfinder.PathFindingQUBOGeneratorSettings): The settings of the QUBO generator.

        Returns:
            sp.Expr: An expression representing the cost function as a QUBO function.
        """
        return self.get_formula_general(graph, settings, FormulaHelpers.get_encoding_variable_binary)

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"{self.__class__.__name__}"


class CompositeCostFunction(CostFunction):
    """A composite cost function that is the sum of multiple cost functions.

    Attributes:
        summands (list[tuple[CostFunction, int]]): A list of tuples of cost functions and their weights.
    """

    summands: list[tuple[CostFunction, int]]

    def __init__(self, *parts: tuple[CostFunction, int]) -> None:
        """Initialises a composite cost function.

        Args:
            *parts (tuple[CostFunction, int]): A list of tuples of cost functions and their weights.
        """
        self.summands = list(parts)

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return "   " + "\n + ".join([f"{w} * {fn}" for (fn, w) in self.summands])

    @override
    def get_formula(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return functools.reduce(
            lambda a, b: a + b[1] * b[0].get_formula(graph, settings),
            self.summands[1:],
            self.summands[0][1] * self.summands[0][0].get_formula(graph, settings),
        )

    @override
    def get_formula_general(
        self,
        _graph: Graph,
        _settings: pathfinder.PathFindingQUBOGeneratorSettings,
        _get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        msg = "This method should not be called for a composite cost function."
        raise RuntimeError(msg)


class PathPositionIs(CostFunction):
    """A cost function that penalises paths that do not contain a given vertex at a given position.

    Attributes:
        vertex_ids (list[int]): The list of vertices, one of which must be located at the given position.
        path (int): The path index.
        position (int): The position index.
    """

    vertex_ids: list[int]
    path: int
    position: int

    def __init__(self, position: int, vertex_ids: list[int], path: int) -> None:
        """Initialises a PathPositionIs cost function.

        Args:
            position (int): The position index.
            vertex_ids (list[int]): The list of vertices, one of which must be located at the given position.
            path (int): The path index.
        """
        self.vertex_ids = vertex_ids
        self.position = position
        self.path = path

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return (
            1
            - FormulaHelpers.sum_set(
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
        ) ** 2

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"PathPosition[{self.position}]Is[{','.join([str(v) for v in self.vertex_ids])}]"


class PathStartsAt(PathPositionIs):
    """A cost function that penalises paths that do not start at a given vertex."""

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        """Initialises a PathStartsAt cost function.

        Args:
            vertex_ids (list[int]): The list of vertices, one of which must be located at the start of the path.
            path (int): The path index.
        """
        super().__init__(1, vertex_ids, path)

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"PathStartsAt[{','.join([str(v) for v in self.vertex_ids])}]"


class PathEndsAt(CostFunction):
    """A cost function that penalises paths that do not end at a given vertex.

    Attributes:
        vertex_ids (list[int]): The list of vertices, one of which must be located at the end of the path.
        path (int): The path index.
    """

    vertex_ids: list[int]
    path: int

    def __init__(self, vertex_ids: list[int], path: int) -> None:
        """Initialises a PathEndsAt cost function.

        Args:
            vertex_ids (list[int]): The list of vertices, one of which must be located at the end of the path.
            path (int): The path index.
        """
        self.vertex_ids = vertex_ids
        self.path = path

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.sum_from_to(
            (
                1
                - FormulaHelpers.get_for_each_vertex(
                    get_variable_function(self.path, "v", "i", graph.n_vertices), graph.all_vertices
                )
            )
            ** 2
            * FormulaHelpers.sum_set(
                get_variable_function(self.path, "v", FormulaHelpers.variable("i") - 1, graph.n_vertices),
                ["v"],
                f"\\not \\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                lambda: list(set(graph.all_vertices) - set(self.vertex_ids)),
            ),
            "i",
            2,
            settings.max_path_length,
        ) + FormulaHelpers.sum_set(
            get_variable_function(self.path, "v", settings.max_path_length, graph.n_vertices),
            ["v"],
            f"\\not \\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
            lambda: list(set(graph.all_vertices) - set(self.vertex_ids)),
        )

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"PathEndsAt[{','.join([str(v) for v in self.vertex_ids])}]"


class PathContainsVertices(CostFunction):
    """A cost function that penalises paths that do not contain a given set of vertices.

    Attributes:
        vertex_ids (list[int]): The list of vertices subject to the constraint.
        min_occurrences (int): The minimum number of occurrences of the vertices in the path.
        max_occurrences (int): The maximum number of occurrences of the vertices in the path.
        path_ids (list[int]): The list of paths to which the cost function applies.
    """

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
        """Initialises a PathContainsVertices cost function.

        Args:
            min_occurrences (int): The minimum number of occurrences of the vertices in the path.
            max_occurrences (int): The maximum number of occurrences of the vertices in the path.
            vertex_ids (list[int]): The list of vertices subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        self.vertex_ids = vertex_ids
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.path_ids = path_ids

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        vertices = ",".join([str(v) for v in self.vertex_ids])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"

    def _handle_for_each(self, expression: sp.Expr) -> sp.Expr:
        """Wraps an expression in the sum parts that are required for this constraint.

        Args:
            expression (sp.Expr): The expression to wrap.

        Returns:
            sp.Expr: The wrapped expression.
        """
        return FormulaHelpers.get_for_each_path(
            (
                FormulaHelpers.sum_set(
                    expression,
                    ["v"],
                    f"\\in \\left\\{{ {', '.join([str(v) for v in self.vertex_ids])} \\right\\}}",
                    lambda: list(self.vertex_ids),
                )
            ),
            self.path_ids,
        )


class PathContainsVerticesExactlyOnce(PathContainsVertices):
    """A cost function that penalises paths that do not contain a given set of vertices exactly once."""

    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        """Initialises a PathContainsVerticesExactlyOnce cost function.

        Args:
            vertex_ids (list[int]): The list of vertices subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(1, 1, vertex_ids, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            (
                1
                - FormulaHelpers.get_for_each_position(
                    get_variable_function("p", "v", "i", graph.n_vertices), settings.max_path_length
                )
            )
            ** 2
        )


class PathContainsVerticesAtLeastOnce(PathContainsVertices):
    """A cost function that penalises paths that do not contain a given set of vertices at least once."""

    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        """Initialises a PathContainsVerticesAtLeastOnce cost function.

        Args:
            vertex_ids (list[int]): The list of vertices subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(1, -1, vertex_ids, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            FormulaHelpers.prod_from_to(
                (1 - get_variable_function("p", "v", "i", graph.n_vertices)), "i", 1, settings.max_path_length
            )
        )


class PathContainsVerticesAtMostOnce(PathContainsVertices):
    """A cost function that penalises paths that do not contain a given set of vertices at most once."""

    def __init__(self, vertex_ids: list[int], path_ids: list[int]) -> None:
        """Initialises a PathContainsVerticesAtMostOnce cost function.

        Args:
            vertex_ids (list[int]): The list of vertices subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(0, 1, vertex_ids, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            FormulaHelpers.sum_from_to(
                FormulaHelpers.sum_from_to(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "v", "j", graph.n_vertices),
                    "j",
                    FormulaHelpers.variable("i") + 1,
                    settings.max_path_length,
                ),
                "i",
                1,
                settings.max_path_length - 1,
            )
        )


class PathContainsEdges(CostFunction):
    """A cost function that penalises paths that do not contain a given set of edges.

    Attributes:
        edges (list[tuple[int, int]]): The list of edges subject to the constraint.
        min_occurrences (int): The minimum number of occurrences of the edges in the path.
        max_occurrences (int): The maximum number of occurrences of the edges in the path.
        path_ids (list[int]): The list of paths to which the cost function applies.
    """

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
        """Initialises a PathContainsEdges cost function.

        Args:
            min_occurrences (int): The minimum number of occurrences of the edges in the path.
            max_occurrences (int): The maximum number of occurrences of the edges in the path.
            edges (list[tuple[int, int]]): The list of edges subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        self.edges = edges
        self.min_occurrences = min_occurrences
        self.max_occurrences = max_occurrences
        self.path_ids = path_ids

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        vertices = ",".join([str(v) for v in self.edges])
        return f"PathContains[{vertices}]:[{self.min_occurrences}-{self.max_occurrences}]"

    def _handle_for_each(self, expression: sp.Expr) -> sp.Expr:
        """Wraps an expression in the sum parts that are required for this constraint.

        Args:
            expression (sp.Expr): The expression to wrap.

        Returns:
            sp.Expr: The wrapped expression.
        """
        return FormulaHelpers.get_for_each_path(
            FormulaHelpers.sum_set(
                expression,
                ["v", "w"],
                f"\\in \\left\\{{ {', '.join(['(' + str(v) + ', ' + str(w) + ')' for (v, w) in self.edges])} \\right\\}}",
                lambda: list(self.edges),
            ),
            self.path_ids,
        )


class PathContainsEdgesExactlyOnce(PathContainsEdges):
    """A cost function that penalises paths that do not contain a given set of edges exactly once."""

    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        """Initialises a PathContainsEdgesExactlyOnce cost function.

        Args:
            edges (list[tuple[int, int]]): The list of edges subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(1, 1, edges, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            (
                1
                - FormulaHelpers.sum_from_to(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    "i",
                    1,
                    settings.max_path_length,
                )
            )
            ** 2
        )


class PathContainsEdgesAtLeastOnce(PathContainsEdges):
    """A cost function that penalises paths that do not contain a given set of edges at least once."""

    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        """Initialises a PathContainsEdgesAtLeastOnce cost function.

        Args:
            edges (list[tuple[int, int]]): The list of edges subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(1, -1, edges, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            FormulaHelpers.prod_from_to(
                (
                    1
                    - get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            )
        )


class PathContainsEdgesAtMostOnce(PathContainsEdges):
    """A cost function that penalises paths that do not contain a given set of edges at most once."""

    def __init__(self, edges: list[tuple[int, int]], path_ids: list[int]) -> None:
        """Initialises a PathContainsEdgesAtMostOnce cost function.

        Args:
            edges (list[tuple[int, int]]): The list of edges subject to the constraint.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(0, 1, edges, path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return self._handle_for_each(
            FormulaHelpers.sum_from_to(
                FormulaHelpers.sum_from_to(
                    (
                        get_variable_function("p", "v", "i", graph.n_vertices)
                        * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices)
                        * get_variable_function("p", "v", "j", graph.n_vertices)
                        * get_variable_function("p", "w", FormulaHelpers.variable("j") + 1, graph.n_vertices)
                    ),
                    "j",
                    FormulaHelpers.variable("i") + 1,
                    settings.max_path_length,
                ),
                "i",
                1,
                settings.max_path_length - 1,
            )
        )


class PathBound(CostFunction):
    """An abstract cost function for penalties limited to an individual path.

    Attributes:
        path_ids (list[int]): The list of paths to which the cost function applies.
    """

    path_ids: list[int]

    def __init__(self, path_ids: list[int]) -> None:
        """Initialises a PathBound cost function.

        Args:
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        self.path_ids = path_ids

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"{self.__class__.__name__}[{','.join([str(path_id) for path_id in self.path_ids])}]"


class PrecedenceConstraint(PathBound):
    """A cost function that penalises paths that do not satisfy a given precedence constraint.

    Attributes:
        pre (int): The vertex that must precede `post`.
        post (int): The vertex that must follow `pre`.
    """

    pre: int
    post: int

    def __init__(self, pre: int, post: int, path_ids: list[int]) -> None:
        """Initialises a PrecedenceConstraint cost function.

        Args:
            pre (int): The vertex that must precede `post`.
            post (int): The vertex that must follow `pre`.
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(path_ids)
        self.pre = pre
        self.post = post

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.get_for_each_path(
            FormulaHelpers.sum_from_to(
                get_variable_function("p", self.post, "i", graph.n_vertices)
                * FormulaHelpers.prod_from_to(
                    (1 - get_variable_function("p", self.pre, "j", graph.n_vertices)),
                    "j",
                    1,
                    FormulaHelpers.variable("i") - 1,
                ),
                "i",
                1,
                settings.max_path_length,
            ),
            self.path_ids,
        )


class PathComparison(CostFunction):
    """An abstract cost function for penalties that compare two paths.

    Attributes:
        path_one (int): The first path.
    path_two (int): The second path.
    """

    path_one: int
    path_two: int

    def __init__(self, path_one: int, path_two: int) -> None:
        """Initialises a PathComparison cost function.

        Args:
            path_one (int): The first path.
            path_two (int): The second path.
        """
        self.path_one = path_one
        self.path_two = path_two

    def __str__(self) -> str:
        """Returns a string representation of the cost function.

        Returns:
            str: A string representation of the cost function.
        """
        return f"{self.__class__.__name__}[{self.path_one}, {self.path_two}]"


class PathsShareNoVertices(PathComparison):
    """A cost function that penalises paths that share vertices."""

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.get_for_each_vertex(
            FormulaHelpers.get_for_each_position(
                get_variable_function(self.path_one, "v", "i", graph.n_vertices), settings.max_path_length
            )
            * FormulaHelpers.get_for_each_position(
                get_variable_function(self.path_two, "v", "i", graph.n_vertices), settings.max_path_length
            ),
            graph.all_vertices,
        )


class PathsShareNoEdges(PathComparison):
    """A cost function that penalises paths that share edges."""

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.sum_set(
            FormulaHelpers.sum_from_to(
                (
                    get_variable_function(self.path_one, "v", "i", graph.n_vertices)
                    * get_variable_function(self.path_one, "w", FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            )
            * FormulaHelpers.sum_from_to(
                (
                    get_variable_function(self.path_two, "v", "i", graph.n_vertices)
                    * get_variable_function(self.path_two, "w", FormulaHelpers.variable("i") + 1, graph.n_vertices)
                ),
                "i",
                1,
                settings.max_path_length,
            ),
            ["v", "w"],
            "\\in E",
            lambda: cast(list[Union[sp.Expr, int, float, tuple[Union[sp.Expr, int, float], ...]]], graph.all_edges),
        )


class PathIsValid(PathBound):
    """A cost function that penalises paths that are not valid for a given encoding.

    A path may be invalid if
        - it contains an edge that is not in the graph,
        - it has an assignment incompatible with the encoding:
            - for One-Hot encoding, multiple vertices are assigned to a single position
            - for Domain-Wall encoding, a position bit string is of the form `1...10...1...0`
    """

    def __init__(self, path_ids: list[int]) -> None:
        """Initialises a PathIsValid cost function.

        Args:
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.get_for_each_path(
            FormulaHelpers.sum_set(
                FormulaHelpers.get_for_each_position(
                    get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    settings.max_path_length if settings.loops else settings.max_path_length - 1,
                ),
                ["v", "w"],
                "\\not\\in E",
                lambda: cast(list[Union[sp.Expr, int, float, tuple[Union[sp.Expr, int, float], ...]]], graph.non_edges),
            ),
            self.path_ids,
        )

    @override
    def get_formula_one_hot(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        def get_variable_function(p: Any, v: Any, i: Any, _n: int = 0) -> sp.Expr:
            return FormulaHelpers.get_encoding_variable_one_hot(p, v, i)

        general = self.get_formula_general(graph, settings, get_variable_function)
        return general + FormulaHelpers.get_for_each_path(
            FormulaHelpers.get_for_each_position(
                (1 - FormulaHelpers.get_for_each_vertex(get_variable_function("p", "v", "i"), graph.all_vertices))
                * -1
                * (FormulaHelpers.get_for_each_vertex(get_variable_function("p", "v", "i"), graph.all_vertices)),
                settings.max_path_length,
            ),
            self.path_ids,
        )

    @override
    def get_formula_domain_wall(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        general = self.get_formula_general(graph, settings, FormulaHelpers.get_encoding_variable_domain_wall)
        enforce_domain_wall_penalty: sp.Expr = (
            2 * settings.max_path_length * np.max(graph.adjacency_matrix) + graph.n_vertices**2
        )
        # This ensures that the domain wall condition (x_i = 0 -> x_{i+1} = 0) is not broken to achieve better cost in other cost functions.
        return general + enforce_domain_wall_penalty * FormulaHelpers.get_for_each_path(
            FormulaHelpers.get_for_each_position(
                FormulaHelpers.sum_set(
                    (1 - FormulaHelpers.get_encoding_variable_one_hot("p", "v", "i"))
                    * FormulaHelpers.get_encoding_variable_one_hot("p", FormulaHelpers.variable("v") + 1, "i"),
                    ["v"],
                    "\\in V",
                    cast(SetCallback, lambda: graph.all_vertices),
                ),
                settings.max_path_length,
            ),
            self.path_ids,
        )

    @override
    def get_formula_binary(self, graph: Graph, settings: pathfinder.PathFindingQUBOGeneratorSettings) -> sp.Expr:
        return self.get_formula_general(graph, settings, FormulaHelpers.get_encoding_variable_binary)


class MinimizePathLength(PathBound):
    """A cost function that penalises paths based on their length.

    A bigger total weight causes a bigger penalty.
    """

    def __init__(self, path_ids: list[int]) -> None:
        """Initialises a MinimizePathLength cost function.

        Args:
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return FormulaHelpers.get_for_each_path(
            FormulaHelpers.sum_set(
                FormulaHelpers.get_for_each_position(
                    FormulaHelpers.adjacency("v", "w")
                    * get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    settings.max_path_length,
                ),
                ["v", "w"],
                "\\in E",
                lambda: cast(list[Union[sp.Expr, int, float, tuple[Union[sp.Expr, int, float], ...]]], graph.all_edges),
            ),
            self.path_ids,
        )


class MaximizePathLength(PathBound):
    """A cost function that penalises paths based on their length.

    A lower total weight causes a bigger penalty.
    """

    def __init__(self, path_ids: list[int]) -> None:
        """Initialises a MaximizePathLength cost function.

        Args:
            path_ids (list[int]): The list of paths to which the cost function applies.
        """
        super().__init__(path_ids)

    @override
    def get_formula_general(
        self,
        graph: Graph,
        settings: pathfinder.PathFindingQUBOGeneratorSettings,
        get_variable_function: GetVariableFunction,
    ) -> sp.Expr:
        return -1 * FormulaHelpers.get_for_each_path(
            FormulaHelpers.sum_set(
                FormulaHelpers.get_for_each_position(
                    FormulaHelpers.adjacency("v", "w")
                    * get_variable_function("p", "v", "i", graph.n_vertices)
                    * get_variable_function("p", "w", FormulaHelpers.variable("i") + 1, graph.n_vertices),
                    settings.max_path_length,
                ),
                ["v", "w"],
                "\\in E",
                lambda: cast(list[Union[sp.Expr, int, float, tuple[Union[sp.Expr, int, float], ...]]], graph.all_edges),
            ),
            self.path_ids,
        )


def merge(cost_functions: list[CostFunction], optimisation_goals: list[CostFunction]) -> CompositeCostFunction:
    """Merges a list of cost functions and a list of optimisation criteria into a single cost function.

    Args:
        cost_functions (list[CostFunction]): The list of cost functions.
        optimisation_goals (list[CostFunction]): The optimisation criteria.

    Returns:
        CompositeCostFunction: The resulting cost function.
    """
    return CompositeCostFunction(*([(f, 1) for f in cost_functions] + [(f, 1) for f in optimisation_goals]))
