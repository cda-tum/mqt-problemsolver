"""This module is responsible for the pathfinding version of the QUBOGenerator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources as impresources
from typing import TYPE_CHECKING, Any, Sequence, cast

import jsonschema
import numpy as np
import sympy as sp
from typing_extensions import override

from mqt.qubomaker import qubo_generator
from mqt.qubomaker.pathfinder import cost_functions as cf

if TYPE_CHECKING:
    from mqt.qubomaker.graph import Graph


@dataclass
class PathFindingQUBOGeneratorSettings:
    """A dataclass containing the settings for the pathfinding QUBO generator."""

    encoding_type: cf.EncodingType
    n_paths: int
    max_path_length: int
    loops: bool = False


class PathFindingQUBOGenerator(qubo_generator.QUBOGenerator):
    """A class for generating QUBOs for pathfinding problems.

    Extends the QUBOGenerator class with methods for generating QUBOs for pathfinding problems.

    Attributes:
        graph (Graph): The graph on which the pathfinding problem is defined.
        settings (PathFindingQUBOGeneratorSettings): The settings for the QUBO generator.
    """

    graph: Graph
    settings: PathFindingQUBOGeneratorSettings

    def __init__(
        self,
        objective_function: cf.CostFunction | None,
        graph: Graph,
        settings: PathFindingQUBOGeneratorSettings,
    ) -> None:
        """Initialises a PathFindingQUBOGenerator object.

        Args:
            objective_function (cf.CostFunction | None): The objective function of the pathfinding problem.
            graph (Graph): The graph on which the pathfinding problem is defined.
            settings (PathFindingQUBOGeneratorSettings): The settings for the QUBO generator.
        """
        super().__init__(objective_function.get_formula(graph, settings) if objective_function is not None else None)
        self.graph = graph
        self.settings = settings

    @staticmethod
    def suggest_encoding(json_string: str, graph: Graph) -> cf.EncodingType:
        """Suggests an encoding type for a given pathfinding problem.

        The suggestion is based on the number of binary variables required for a specific problem.

        Args:
            json_string (str): A JSON string describing the pathfinding problem.
            graph (Graph): The graph on which the pathfinding problem is defined.

        Returns:
            cf.EncodingType: The suggested encoding type.
        """
        results: list[tuple[cf.EncodingType, int]] = []
        for encoding in [cf.EncodingType.ONE_HOT, cf.EncodingType.UNARY, cf.EncodingType.BINARY]:
            generator = PathFindingQUBOGenerator.__from_json(json_string, graph, override_encoding=encoding)
            results.append((encoding, generator.construct_qubo_matrix().shape[0]))
        return next(encoding for (encoding, size) in results if size == min([size for (_, size) in results]))

    @staticmethod
    def from_json(json_string: str, graph: Graph) -> PathFindingQUBOGenerator:
        """Creates a PathFindingQUBOGenerator object from its JSON format.

        Args:
            json_string (str): The JSON string describing the pathfinding problem.
            graph (Graph): The graph on which the pathfinding problem is defined.

        Returns:
            PathFindingQUBOGenerator: The constructed QUBO generator.
        """
        return PathFindingQUBOGenerator.__from_json(json_string, graph)

    @staticmethod
    def __from_json(
        json_string: str, graph: Graph, override_encoding: cf.EncodingType | None = None
    ) -> PathFindingQUBOGenerator:
        """Creates a PathFindingQUBOGenerator object from its JSON format.

        The override_encoding parameter can be used to override the encoding type specified in the JSON string.

        Args:
            json_string (str): The JSON string describing the pathfinding problem.
            graph (Graph): The graph on which the pathfinding problem is defined.
            override_encoding (cf.EncodingType | None, optional): Can be used to override the encoding type specified in the JSON string. Defaults to None.

        Raises:
            ValueError: If any of the constraints in the JSON string is not supported.

        Returns:
            PathFindingQUBOGenerator: The constructed QUBO generator.
        """
        with (impresources.files(__package__) / "resources" / "input-format.json").open("r") as f:
            main_schema = json.load(f)
        with (impresources.files(__package__) / "resources" / "constraint.json").open("r") as f:
            constraint_schema = json.load(f)
        resolver = jsonschema.RefResolver.from_schema(main_schema)
        resolver.store["constraint.json"] = constraint_schema
        for file in os.listdir(str(impresources.files(__package__) / "resources" / "constraints")):
            with (impresources.files(__package__) / "resources" / "constraints" / file).open("r") as f:
                resolver.store[file] = json.load(f)

        validator = jsonschema.Draft7Validator(main_schema, resolver=resolver)
        json_object = json.loads(json_string)
        validator.validate(json_object)

        if override_encoding is None:
            if json_object["settings"]["encoding"] == "ONE_HOT":
                encoding_type = cf.EncodingType.ONE_HOT
            elif json_object["settings"]["encoding"] in ["UNARY", "DOMAIN_WALL"]:
                encoding_type = cf.EncodingType.UNARY
            else:
                encoding_type = cf.EncodingType.BINARY
        else:
            encoding_type = override_encoding

        settings = PathFindingQUBOGeneratorSettings(
            encoding_type,
            json_object["settings"].get("n_paths", 1),
            json_object["settings"].get("max_path_length", 0),
            json_object["settings"].get("loops", False),
        )
        if settings.max_path_length == 0:
            settings.max_path_length = graph.n_vertices

        def get_constraint(constraint: dict[str, Any]) -> list[cf.CostFunction]:
            if constraint["type"] == "PathIsValid":
                return [cf.PathIsValid(constraint.get("path_ids", [1]))]
            if constraint["type"] == "MinimisePathLength":
                return [cf.MinimisePathLength(constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathStartsAt":
                return [cf.PathStartsAt(constraint["vertices"], constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathEndsAt":
                return [cf.PathEndsAt(constraint["vertices"], constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathPositionIs":
                return [
                    cf.PathPositionIs(constraint["position"], constraint["vertices"], constraint.get("path_ids", [1]))
                ]
            if constraint["type"] == "PathContainsVerticesExactlyOnce":
                vertices = constraint.get("vertices", [])
                if len(vertices) == 0:
                    vertices = graph.all_vertices
                return [cf.PathContainsVerticesExactlyOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsVerticesAtLeastOnce":
                vertices = constraint.get("vertices", [])
                if len(vertices) == 0:
                    vertices = graph.all_vertices
                return [cf.PathContainsVerticesAtLeastOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsVerticesAtMostOnce":
                vertices = constraint.get("vertices", [])
                if len(vertices) == 0:
                    vertices = graph.all_vertices
                return [cf.PathContainsVerticesAtMostOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesExactlyOnce":
                edges = [tuple(edge) for edge in constraint.get("edges", [])]
                if len(edges) == 0:
                    edges = graph.all_edges
                return [cf.PathContainsEdgesExactlyOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesAtLeastOnce":
                edges = [tuple(edge) for edge in constraint.get("edges", [])]
                if len(edges) == 0:
                    edges = graph.all_edges
                return [cf.PathContainsEdgesAtLeastOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesAtMostOnce":
                edges = [tuple(edge) for edge in constraint.get("edges", [])]
                if len(edges) == 0:
                    edges = graph.all_edges
                return [cf.PathContainsEdgesAtMostOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PrecedenceConstraint":
                return [
                    cf.PrecedenceConstraint(precedence["before"], precedence["after"], constraint.get("path_ids", [1]))
                    for precedence in constraint["precedences"]
                ]
            if constraint["type"] == "PathsShareNoVertices":
                paths = constraint.get("path_ids", [1])
                return [(cf.PathsShareNoVertices(i, j)) for i in paths for j in paths if i != j]
            if constraint["type"] == "PathsShareNoEdges":
                paths = constraint.get("path_ids", [1])
                return [(cf.PathsShareNoEdges(i, j)) for i in paths for j in paths if i != j]
            msg = f"Constraint {constraint['type']} not supported."
            raise ValueError(msg)

        generator = PathFindingQUBOGenerator(
            get_constraint(json_object["objective_function"])[0] if "objective_function" in json_object else None,
            graph,
            settings,
        )
        if "constraints" in json_object:
            for constraint in json_object["constraints"]:
                get_constraint(constraint)
                for cost_function in get_constraint(constraint):
                    generator.add_constraint(cost_function)

        return generator

    def add_constraint(self, constraint: cf.CostFunction) -> PathFindingQUBOGenerator:
        """Add a pathfinding constraint to the QUBO generator.

        Args:
            constraint (cf.CostFunction): The constraint to be added.

        Returns:
            PathFindingQUBOGenerator: The current instance of the QUBO generator.
        """
        self.add_penalty(constraint.get_formula(self.graph, self.settings))
        return self

    @override
    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        """Compute the suggested lambda values for all penalties.

        Returns:
            list[tuple[sp.Expr, float]]: A list of tuples containing the penalty expressions and their suggested lambda values.
        """
        return [(expr, lam if lam else self.__optimal_lambda()) for (expr, lam) in self.penalties]

    def __optimal_lambda(self) -> float:
        """Compute the optimal lambda value for all penalties."""
        return cast(float, np.max(self.graph.adjacency_matrix) * self.settings.max_path_length + 1)

    @override
    def _construct_expansion(self, expression: sp.Expr) -> sp.Expr:
        """Performs pathfinding-specific tasks during the expansion construction procedure.

        These tasks include the assignment of specific variables, such as adjacency matrix values etc.

        Args:
            expression (sp.Expr): The expression to transform.

        Returns:
            sp.Expr: The transformed expression."""
        assignment = [
            (cf._FormulaHelpers.adjacency(i + 1, j + 1), self.graph.adjacency_matrix[i, j])
            for i in range(self.graph.n_vertices)
            for j in range(self.graph.n_vertices)
        ]
        assignment += [
            (cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, self.graph.n_vertices + 1, i + 1), 0)
            for p in range(self.settings.n_paths)
            for i in range(self.settings.max_path_length + 1)
        ]  # x_{p, |V| + 1, i} = 0 for all p, i
        assignment += [
            (
                cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, self.settings.max_path_length + 1),
                cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, 1) if self.settings.loops else 0,
            )
            for p in range(self.settings.n_paths)
            for v in range(self.graph.n_vertices)
        ]  # x_{p, v, N + 1} = x_{p, v, 1} for all p, v if loop, otherwise 0
        result = expression.subs(dict(assignment))  # type: ignore[no-untyped-call]
        if isinstance(result, sp.Expr):
            return result
        msg = "Expression is not a sympy expression."
        raise ValueError(msg)

    @override
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

    @override
    def decode_bit_array(self, _array: list[int]) -> Any:
        """Given an assignment, decodes it into a meaningful result. May be extended by subclasses.

        Args:
            _array (list[int]): The binary assignment.

        Returns:
            Any: The decoded result as a (set of) path(s).
        """
        if self.settings.encoding_type == cf.EncodingType.ONE_HOT:
            return self.decode_bit_array_one_hot(_array)
        if self.settings.encoding_type == cf.EncodingType.UNARY:
            return self.decode_bit_array_unary(_array)
        if self.settings.encoding_type == cf.EncodingType.BINARY:
            return self.decode_bit_array_binary(_array)
        msg = f"Encoding type {self.settings.encoding_type} not supported."  # type: ignore[unreachable]
        raise ValueError(msg)

    def decode_bit_array_unary(self, array: list[int]) -> Any:
        """Decodes an assignment for unary encoding.

        Args:
            array (list[int]): The assignment to decode.

        Returns:
            Any: The decoded assignment as a (set of) path(s).
        """
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
        """Decodes an assignment for One-Hot encoding.

        Args:
            array (list[int]): The assignment to decode.

        Returns:
            Any: The decoded assignment as a (set of) path(s).
        """
        path = []
        for i, bit in enumerate(array):
            if bit == 0:
                continue
            v = i % self.graph.n_vertices
            s = i // self.graph.n_vertices
            path.append((v, s))
        path.sort(key=lambda x: x[1])
        return [entry[0] + 1 for entry in path]

    def decode_bit_array_binary(self, _array: list[int]) -> Any:
        """Decodes an assignment for Binary encoding.

        Args:
            array (list[int]): The assignment to decode.

        Returns:
            Any: The decoded assignment as a (set of) path(s).
        """
        msg = "TODO"
        raise NotImplementedError(msg)

    @override
    def _get_all_variables(self) -> Sequence[tuple[sp.Expr, int]]:
        result = []
        for p in range(self.settings.n_paths):
            for v in self.graph.all_vertices:
                for i in range(self.settings.max_path_length):
                    var = cf._FormulaHelpers.get_encoding_variable_one_hot(p + 1, v, i + 1)
                    result.append((var, self.get_variable_index(var)))
        return result
