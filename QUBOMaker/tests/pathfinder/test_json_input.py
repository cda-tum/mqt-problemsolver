"""Tests the correctness of the JSON input format."""

from __future__ import annotations

from pathlib import Path

import pytest

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf

from .utils_test import check_equal, get_test_graph

TEST_GRAPH = get_test_graph()


def read_from_path(path: str) -> pf.PathFindingQUBOGenerator:
    """Reads a JSON input file and returns the corresponding `PathFindingQUBOGenerator`.

    Args:
        path (str): The path to the JSON input file.

    Returns:
        pf.PathFindingQUBOGenerator: The corresponding `PathFindingQUBOGenerator`.
    """
    with Path.open(Path("tests") / "pathfinder" / "resources" / "json" / path) as file:
        return pf.PathFindingQUBOGenerator.from_json(file.read(), TEST_GRAPH)


def test_all_constraints() -> None:
    """Tests a JSON input file that includes all constraints."""
    json_generator = read_from_path("all.json")

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=3,
        max_path_length=4,
        loops=True,
    )
    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MinimizePathLength([1]), graph=TEST_GRAPH, settings=settings
    )

    manual_generator.add_constraint(cf.PathContainsEdgesAtLeastOnce([(1, 2)], [1]))
    manual_generator.add_constraint(cf.PathContainsEdgesAtMostOnce([(1, 2)], [1]))
    manual_generator.add_constraint(cf.PathContainsEdgesExactlyOnce([(1, 2)], [1]))
    manual_generator.add_constraint(cf.PathContainsVerticesAtLeastOnce([1, 2, 3], [1]))
    manual_generator.add_constraint(cf.PathContainsVerticesAtMostOnce([1, 2, 3], [1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce([1, 2, 3], [1]))
    manual_generator.add_constraint(cf.PathEndsAt([1, 2, 3], 1))
    manual_generator.add_constraint(cf.PathStartsAt([1, 2, 3], 1))
    manual_generator.add_constraint(cf.PathPositionIs(2, [1, 2, 3], 1))
    manual_generator.add_constraint(cf.PrecedenceConstraint(1, 2, [1]))
    manual_generator.add_constraint(cf.PathsShareNoEdges(1, 2))
    manual_generator.add_constraint(cf.PathsShareNoEdges(2, 3))
    manual_generator.add_constraint(cf.PathsShareNoEdges(1, 3))
    manual_generator.add_constraint(cf.PathsShareNoVertices(1, 2))
    manual_generator.add_constraint(cf.PathsShareNoVertices(2, 3))
    manual_generator.add_constraint(cf.PathsShareNoVertices(1, 3))
    manual_generator.add_constraint(cf.PathIsValid([1, 2, 3]))

    check_equal(json_generator, manual_generator)


def test_alternative_options() -> None:
    """Tests a JSON input file that includes alternative (non-default) options."""
    json_generator = read_from_path("alternative_options.json")

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.BINARY,
        n_paths=2,
        max_path_length=5,
        loops=False,
    )
    manual_generator = pf.PathFindingQUBOGenerator(objective_function=None, graph=TEST_GRAPH, settings=settings)

    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(TEST_GRAPH.all_vertices, [1]))
    manual_generator.add_constraint(cf.PathContainsVerticesAtLeastOnce([1, 2], [1]))
    manual_generator.add_constraint(cf.PathContainsVerticesAtMostOnce(TEST_GRAPH.all_vertices, [1, 2]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(1, 2, [1]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(2, 3, [1]))

    check_equal(json_generator, manual_generator)


def test_suggest_encoding() -> None:
    """Tests the encoding suggestion feature for a JSON input file."""
    with Path.open(Path("tests") / "pathfinder" / "resources" / "json" / "with_weight.json") as file:
        j = file.read()
    assert pf.PathFindingQUBOGenerator.suggest_encoding(j, TEST_GRAPH) == pf.EncodingType.ONE_HOT


def test_with_weight() -> None:
    """Tests a JSON input file that includes weights for some constraints."""
    json_generator = read_from_path("with_weight.json")

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.DOMAIN_WALL,
        n_paths=1,
        max_path_length=5,
        loops=False,
    )
    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MaximizePathLength([1]), graph=TEST_GRAPH, settings=settings
    )

    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(TEST_GRAPH.all_vertices, [1]), weight=500)

    check_equal(json_generator, manual_generator)


def test_fail_excess_field() -> None:
    """Tests a JSON input file that should fail because it includes an excess field."""
    with pytest.raises(ValueError, match="JSON"):
        read_from_path("fail/excess_field.json")


def test_fail_missing_field() -> None:
    """Tests a JSON input file that should fail because it is missing a field."""
    with pytest.raises(ValueError, match="JSON"):
        read_from_path("fail/missing_field.json")


def test_fail_too_few_elements() -> None:
    """Tests a JSON input file that should fail because some options have too few elements."""
    with pytest.raises(ValueError, match="JSON"):
        read_from_path("fail/too_few_elements.json")


def test_fail_unknown_type() -> None:
    """Tests a JSON input file that should fail because it includes an unknown type."""
    with pytest.raises(ValueError, match="JSON"):
        read_from_path("fail/unknown_type.json")
