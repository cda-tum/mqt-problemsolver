"""Tests the features of the `graph` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mqt.qubomaker import Graph


def test_init_with_edge_list() -> None:
    """Tests the initialization of a `Graph` object with an edge list or an adjacency matrix."""
    g = Graph(5, [(1, 2, 4), (3, 5, 2), (1, 3, 2), (4, 5, 5), (2, 4, 3), (5, 1)])

    adjacency_matrix = [[0, 4, 2, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 5], [1, 0, 0, 0, 0]]

    g2 = Graph.from_adjacency_matrix(adjacency_matrix)

    assert np.array_equal(g.adjacency_matrix, g2.adjacency_matrix)
    assert g.all_edges == g2.all_edges
    assert g.all_vertices == g2.all_vertices
    assert g.non_edges == g2.non_edges
    assert np.array_equal(g.adjacency_matrix, adjacency_matrix)
    assert g == g2


def test_read_write() -> None:
    """Tests the read and write operations of the `Graph` class."""
    g = Graph(5, [(1, 2, 4), (3, 5, 2), (1, 3, 2), (4, 5, 5), (2, 4, 3), (5, 1)])

    with Path.open(Path("tests") / "resources" / "graph" / "graph", "w") as file:
        g.store(file)

    with Path.open(Path("tests") / "resources" / "graph" / "graph", "r") as file:
        g2 = Graph.read(file)

    assert g == g2

    with Path.open(Path("tests") / "resources" / "graph" / "graph", "w") as file:
        file.write(g.serialize())

    with Path.open(Path("tests") / "resources" / "graph" / "graph", "r") as file:
        g2 = Graph.deserialize(file.read())

    assert g == g2


def test_eq() -> None:
    """Tests the equality operator on `Graph` objects."""
    g1 = Graph(3, [(1, 2, 4), (1, 3, 1), (2, 3, 1), (2, 1, 5), (3, 1, 4), (3, 2, 5)])

    g2 = Graph(
        3,
        [
            (3, 2, 5),
            (1, 2, 4),
            (2, 1, 5),
            (2, 3, 1),
            (3, 1, 4),
            (1, 3, 1),
        ],
    )

    g3 = Graph(3, [(1, 2, 4), (1, 3, 1), (2, 3, 1), (3, 1, 4), (3, 2, 5)])

    g4 = Graph(3, [(1, 2, 4), (1, 3, 1), (2, 3, 1), (2, 1, 3), (3, 1, 4), (3, 2, 5)])

    g5 = Graph(3, [(1, 2, 4), (1, 3), (2, 3), (2, 1, 5), (3, 1, 4), (3, 2, 5)])

    assert g1 == g2
    assert g1 == g5
    assert g2 == g5
    assert g1 != g3
    assert g1 != g4
    assert g2 != g3
    assert g2 != g4
    assert g5 != g3
    assert g5 != g4
    assert g3 != g4

    assert g2 == g1
    assert g5 == g1
    assert g5 == g2
    assert g3 != g1
    assert g4 != g1
    assert g3 != g2
    assert g4 != g2
    assert g3 != g5
    assert g4 != g5
    assert g4 != g3
