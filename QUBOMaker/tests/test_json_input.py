from __future__ import annotations

from pathlib import Path

import pytest

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf

from .utils_test import get_test_graph

TEST_GRAPH = get_test_graph()


def read_from_path(path: str) -> pf.PathFindingQUBOGenerator:
    with Path.open(Path("tests") / "resources" / "json" / path) as file:
        return pf.PathFindingQUBOGenerator.from_json(file.read(), TEST_GRAPH)


def check_equal(a: pf.PathFindingQUBOGenerator, b: pf.PathFindingQUBOGenerator) -> None:
    assert a.objective_function == b.objective_function
    assert a.graph == b.graph
    assert a.settings == b.settings

    print(len(a.penalties), len(b.penalties))

    for expr, weight in a.penalties:
        assert len([w for (e, w) in b.penalties if e == expr and w == weight]) == 1

    for expr, weight in b.penalties:
        assert len([w for (e, w) in a.penalties if e == expr and w == weight]) == 1


class TestJsonInput:
    def test_all_constraints(self) -> None:
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

        check_equal(json_generator, manual_generator)

    def test_alternative_options(self) -> None:
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

    def test_with_weight(self) -> None:
        json_generator = read_from_path("with_weight.json")

        settings = pf.PathFindingQUBOGeneratorSettings(
            encoding_type=pf.EncodingType.UNARY,
            n_paths=1,
            max_path_length=5,
            loops=False,
        )
        manual_generator = pf.PathFindingQUBOGenerator(
            objective_function=cf.MaximizePathLength([1]), graph=TEST_GRAPH, settings=settings
        )

        manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(TEST_GRAPH.all_vertices, [1]), weight=500)

        check_equal(json_generator, manual_generator)

    def test_fail_excess_field(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            read_from_path("fail/excess_field.json")

    def test_fail_missing_field(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            read_from_path("fail/missing_field.json")

    def test_fail_too_few_elements(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            read_from_path("fail/too_few_elements.json")

    def test_fail_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            read_from_path("fail/unknown_type.json")
