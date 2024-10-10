"""This module is responsible for the pathfinding version of the QUBOGenerator."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING or sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources

import operator

import numpy as np
import sympy as sp
from jsonschema import Draft7Validator, ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7
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
        for encoding in [cf.EncodingType.ONE_HOT, cf.EncodingType.DOMAIN_WALL, cf.EncodingType.BINARY]:
            generator = PathFindingQUBOGenerator.__from_json(json_string, graph, override_encoding=encoding)
            results.append((encoding, generator.count_required_variables()))
        return next(encoding for (encoding, size) in results if size == min(size for (_, size) in results))

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
        with (resources.files(__package__) / "resources" / "input-format.json").open("r") as f:
            main_schema = json.load(f)
        with (resources.files(__package__) / "resources" / "constraint.json").open("r") as f:
            constraint_schema = json.load(f)

        registry: Registry[dict[str, Any]] = Registry().with_resources([
            ("main_schema", Resource.from_contents(main_schema, DRAFT7)),
            ("constraint.json", Resource.from_contents(constraint_schema, DRAFT7)),
        ])
        constraints_dir = resources.files(__package__) / "resources" / "constraints"
        for file in constraints_dir.iterdir():
            with file.open("r") as f:
                registry = registry.with_resource(file.name, Resource.from_contents(json.load(f), DRAFT7))

        validator = Draft7Validator(main_schema, registry=registry)
        json_object = json.loads(json_string)

        try:
            validator.validate(json_object)
        except ValidationError as e:
            msg = f"Invalid JSON: {e.message}"
            raise ValueError(msg) from e

        if override_encoding is None:
            if json_object["settings"]["encoding"] == "ONE_HOT":
                encoding_type = cf.EncodingType.ONE_HOT
            elif json_object["settings"]["encoding"] in {"UNARY", "DOMAIN_WALL"}:
                encoding_type = cf.EncodingType.DOMAIN_WALL
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

        def get_vertices_possibly_all(constraint: dict[str, Any]) -> list[int]:
            vertices = constraint.get("vertices", [])
            if len(vertices) == 0:
                vertices = graph.all_vertices
            return cast(list[int], vertices)

        def get_edges_possibly_all(constraint: dict[str, Any]) -> list[tuple[int, int]]:
            edges = [tuple(edge) for edge in constraint.get("edges", [])]
            if len(edges) == 0:
                edges = graph.all_edges
            return edges

        def get_constraint(constraint: dict[str, Any]) -> list[cf.CostFunction]:
            if constraint["type"] == "PathIsValid":
                return [cf.PathIsValid(constraint.get("path_ids", [1]))]
            if constraint["type"] == "MinimizePathLength":
                return [cf.MinimizePathLength(constraint.get("path_ids", [1]))]
            if constraint["type"] == "MaximizePathLength":
                return [cf.MaximizePathLength(constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathStartsAt":
                return [cf.PathStartsAt(constraint["vertices"], constraint.get("path_id", 1))]
            if constraint["type"] == "PathEndsAt":
                return [cf.PathEndsAt(constraint["vertices"], constraint.get("path_id", 1))]
            if constraint["type"] == "PathPositionIs":
                return [cf.PathPositionIs(constraint["position"], constraint["vertices"], constraint.get("path_id", 1))]
            if constraint["type"] == "PathContainsVerticesExactlyOnce":
                vertices = get_vertices_possibly_all(constraint)
                return [cf.PathContainsVerticesExactlyOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsVerticesAtLeastOnce":
                vertices = get_vertices_possibly_all(constraint)
                return [cf.PathContainsVerticesAtLeastOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsVerticesAtMostOnce":
                vertices = get_vertices_possibly_all(constraint)
                return [cf.PathContainsVerticesAtMostOnce(vertices, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesExactlyOnce":
                edges = get_edges_possibly_all(constraint)
                return [cf.PathContainsEdgesExactlyOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesAtLeastOnce":
                edges = get_edges_possibly_all(constraint)
                return [cf.PathContainsEdgesAtLeastOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PathContainsEdgesAtMostOnce":
                edges = get_edges_possibly_all(constraint)
                return [cf.PathContainsEdgesAtMostOnce(edges, constraint.get("path_ids", [1]))]
            if constraint["type"] == "PrecedenceConstraint":
                return [
                    cf.PrecedenceConstraint(precedence["before"], precedence["after"], constraint.get("path_ids", [1]))
                    for precedence in constraint["precedences"]
                ]
            if constraint["type"] == "PathsShareNoVertices":
                paths = constraint.get("path_ids", [1])
                return [(cf.PathsShareNoVertices(i, j)) for i in paths for j in paths if i < j]
            if constraint["type"] == "PathsShareNoEdges":
                paths = constraint.get("path_ids", [1])
                return [(cf.PathsShareNoEdges(i, j)) for i in paths for j in paths if i < j]
            msg = f"Constraint {constraint['type']} not supported."
            raise ValueError(msg)

        generator = PathFindingQUBOGenerator(
            get_constraint(json_object["objective_function"])[0] if "objective_function" in json_object else None,
            graph,
            settings,
        )
        if "constraints" in json_object:
            for constraint in json_object["constraints"]:
                weight = constraint.get("weight", None)
                for cost_function in get_constraint(constraint):
                    generator.add_constraint(cost_function, weight=weight)

        return generator

    def add_constraint(self, constraint: cf.CostFunction, weight: float | None = None) -> PathFindingQUBOGenerator:
        """Add a pathfinding constraint to the QUBO generator.

        Args:
            constraint (cf.CostFunction): The constraint to be added.
            weight (float | None, optional): The desired weight of the constraint. Defaults to None.

        Returns:
            PathFindingQUBOGenerator: The current instance of the QUBO generator.
        """
        self.add_penalty(constraint.get_formula(self.graph, self.settings), lam=weight)
        return self

    def add_constraint_if_exists(
        self, constraint: cf.CostFunction | None, weight: float | None = None
    ) -> PathFindingQUBOGenerator:
        """Add a pathfinding constraint to the QUBO generator.

        Args:
            constraint (cf.CostFunction): The constraint to be added.
            weight (float | None, optional): The desired weight of the constraint. Defaults to None.

        Returns:
            PathFindingQUBOGenerator: The current instance of the QUBO generator.
        """
        if constraint is None:
            return self
        return self.add_constraint(constraint, weight)

    @override
    def _select_lambdas(self) -> list[tuple[sp.Expr, float]]:
        """Compute the suggested lambda values for all penalties.

        Returns:
            list[tuple[sp.Expr, float]]: A list of tuples containing the penalty expressions and their suggested lambda values.
        """
        return [(expr, lam or self.__optimal_lambda()) for (expr, lam) in self.penalties]

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
        sp.Expr: The transformed expression.
        """
        assignment = [
            (cf.FormulaHelpers.adjacency(i + 1, j + 1), self.graph.adjacency_matrix[i, j])
            for i in range(self.graph.n_vertices)
            for j in range(self.graph.n_vertices)
        ]
        assignment += [
            (
                cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, self.settings.max_path_length + 1),
                cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v + 1, 1)
                if self.settings.loops
                else sp.Integer(0),
            )
            for p in range(self.settings.n_paths)
            for v in range(self.graph.n_vertices)
        ]  # x_{p, v, N + 1} = x_{p, v, 1} for all p, v if loop, otherwise 0
        assignment += [
            (cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, self.graph.n_vertices + 1, i + 1), 0)
            for p in range(self.settings.n_paths)
            for i in range(self.settings.max_path_length + 1)
        ]  # x_{p, |V| + 1, i} = 0 for all p, i
        return expression.subs(dict(assignment))

    @override
    def get_variable_index(self, var: sp.Expr) -> int:
        parts = var.args

        if any(not isinstance(part, sp.core.Integer) for part in parts):
            msg = "Variable subscripts must be integers."
            raise ValueError(msg)

        max_v = self.graph.n_vertices
        if self.settings.encoding_type == cf.EncodingType.BINARY:
            max_v = int(np.ceil(np.log2(self.graph.n_vertices + 1)))

        p = int(cast(int, parts[0]))
        v = int(cast(int, parts[1]))
        i = int(cast(int, parts[2]))

        return int((v - 1) + (i - 1) * max_v + (p - 1) * self.settings.max_path_length * max_v + 1)

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
        if self.settings.encoding_type == cf.EncodingType.DOMAIN_WALL:
            return self.decode_bit_array_domain_wall(_array)
        if self.settings.encoding_type == cf.EncodingType.BINARY:
            return self.decode_bit_array_binary(_array)
        msg = f"Encoding type {self.settings.encoding_type} not supported."  # type: ignore[unreachable]
        raise ValueError(msg)

    def decode_bit_array_domain_wall(self, array: list[int]) -> list[list[int]]:
        """Decodes an assignment for domain_wall encoding.

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
            paths.append([v for v in path if v != 0])
        return paths

    def decode_bit_array_one_hot(self, array: list[int]) -> list[list[int]]:
        """Decodes an assignment for One-Hot encoding.

        Args:
            array (list[int]): The assignment to decode.

        Returns:
            Any: The decoded assignment as a (set of) path(s).
        """
        paths: list[list[int]] = []
        path: list[tuple[int, int]] = []
        for i, bit in enumerate(array):
            if i % (len(array) / self.settings.n_paths) == 0 and i != 0:
                path.sort(key=operator.itemgetter(1))
                paths.append([entry[0] + 1 for entry in path])
                path = []
            if bit == 0:
                continue
            v = i % self.graph.n_vertices
            s = i // self.graph.n_vertices
            path.append((v, s))

        path.sort(key=operator.itemgetter(1))
        paths.append([entry[0] + 1 for entry in path])

        return paths

    def decode_bit_array_binary(self, array: list[int]) -> list[list[int]]:
        """Decodes an assignment for Binary encoding.

        Args:
            array (list[int]): The assignment to decode.

        Returns:
            Any: The decoded assignment as a (set of) path(s).
        """
        paths = []
        max_v = int(np.ceil(np.log2(self.graph.n_vertices + 1)))
        for p in range(self.settings.n_paths):
            path = []
            for i in range(self.settings.max_path_length):
                v = 0
                for j in range(max_v):
                    v += 2**j * array[j + i * max_v + p * max_v * self.settings.max_path_length]
                path.append(v)
            paths.append([v for v in path if v != 0])
        return paths

    @override
    def _get_encoding_variables(self) -> list[tuple[sp.Expr, int]]:
        result = []
        max_v = self.graph.n_vertices
        if self.settings.encoding_type == cf.EncodingType.BINARY:
            max_v = int(np.ceil(np.log2(self.graph.n_vertices + 1)))
        for p in range(self.settings.n_paths):
            for v in range(1, max_v + 1):
                for i in range(self.settings.max_path_length):
                    var = cf.FormulaHelpers.get_encoding_variable_one_hot(p + 1, v, i + 1)
                    result.append((var, self.get_variable_index(var)))
        return result
