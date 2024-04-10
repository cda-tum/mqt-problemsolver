"""Tests the correctness of the tsplib input format."""

from __future__ import annotations

from pathlib import Path

import pytest
import tsplib95

import mqt.qubomaker.pathfinder as pf
import mqt.qubomaker.pathfinder.cost_functions as cf

from .utils_test import check_equal, get_test_graph

TEST_GRAPH = get_test_graph()


def read_from_path(path: str, encoding: pf.EncodingType = pf.EncodingType.ONE_HOT) -> pf.PathFindingQUBOGenerator:
    """Reads a tsplib input file and returns the corresponding `PathFindingQUBOGenerator`.

    Args:
        path: The path to the tsplib input file.
        encoding: The encoding to use.

    Returns:
        The corresponding `PathFindingQUBOGenerator`.
    """
    pth = Path("tests") / "pathfinder" / "resources" / "tsplib" / path
    problem = tsplib95.load(str(pth))

    return pf.from_tsplib_problem(problem, encoding)


def test_hcp() -> None:
    """Tests a tsplib input file that represents a HCP problem."""
    json_generator = read_from_path("hcp-5.hcp")
    graph = json_generator.graph

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=1,
        max_path_length=5,
        loops=True,
    )

    manual_generator = pf.PathFindingQUBOGenerator(objective_function=None, graph=graph, settings=settings)

    manual_generator.add_constraint(cf.PathIsValid([1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
    check_equal(json_generator, manual_generator)


def test_tsp() -> None:
    """Tests a tsplib input file that represents a TSP problem."""
    json_generator = read_from_path("tsp-5.tsp")
    graph = json_generator.graph

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=1,
        max_path_length=5,
        loops=True,
    )

    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MinimizePathLength([1]), graph=graph, settings=settings
    )

    manual_generator.add_constraint(cf.PathIsValid([1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
    check_equal(json_generator, manual_generator)


def test_atsp() -> None:
    """Tests a tsplib input file that represents an ATSP problem."""
    json_generator = read_from_path("atsp-5.atsp")
    graph = json_generator.graph

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=1,
        max_path_length=5,
        loops=True,
    )

    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MinimizePathLength([1]), graph=graph, settings=settings
    )

    manual_generator.add_constraint(cf.PathIsValid([1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
    check_equal(json_generator, manual_generator)


def test_sop() -> None:
    """Tests a tsplib input file that represents a SOP problem."""
    json_generator = read_from_path("sop-5.sop")
    graph = json_generator.graph

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=1,
        max_path_length=5,
        loops=False,
    )

    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MinimizePathLength([1]), graph=graph, settings=settings
    )

    manual_generator.add_constraint(cf.PathIsValid([1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(1, 2, [1]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(1, 5, [1]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(2, 4, [1]))
    manual_generator.add_constraint(cf.PrecedenceConstraint(2, 5, [1]))
    check_equal(json_generator, manual_generator)


def test_fail_cvrp() -> None:
    """Tests a tsplib input file that represents a CVRP problem. This should fail."""
    with pytest.raises(ValueError, match="CVRP"):
        read_from_path("fail/cvrp-7.vrp")


def test_with_forced_edges() -> None:
    """Tests a tsplib input file that includes forced edges."""
    json_generator = read_from_path("forced-edges.tsp")
    graph = json_generator.graph

    settings = pf.PathFindingQUBOGeneratorSettings(
        encoding_type=pf.EncodingType.ONE_HOT,
        n_paths=1,
        max_path_length=5,
        loops=True,
    )

    manual_generator = pf.PathFindingQUBOGenerator(
        objective_function=cf.MinimizePathLength([1]), graph=graph, settings=settings
    )

    manual_generator.add_constraint(cf.PathIsValid([1]))
    manual_generator.add_constraint(cf.PathContainsVerticesExactlyOnce(graph.all_vertices, [1]))
    manual_generator.add_constraint(cf.PathContainsEdgesExactlyOnce([(1, 5), (5, 3)], [1]))
    check_equal(json_generator, manual_generator)
