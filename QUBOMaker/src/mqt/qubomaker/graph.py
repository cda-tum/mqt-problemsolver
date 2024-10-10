"""Provides a simple implementation for graphs to be used with QUBOMaker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from io import TextIOWrapper

    Edge = Union[tuple[int, int], tuple[int, int, int], tuple[int, int, float]]


class Graph:
    """Represents a graph to be used with QUBOMaker.

    Attributes:
        n_vertices (int): The number of vertices in the graph.
        adjacency_matrix (npt.NDArray[np.int_ | np.float64]): The adjacency matrix of the graph.
        all_vertices (list[int]): A list of all vertices in the graph.
        all_edges (list[tuple[int, int]]): A list of all edges in the graph.
        non_edges (list[tuple[int, int]]): A list of all non-edges in the graph.
    """

    n_vertices: int
    adjacency_matrix: npt.NDArray[np.int_ | np.float64]

    @property
    def all_vertices(self) -> list[int]:
        """A list of all vertices in the graph."""
        return list(range(1, self.n_vertices + 1))

    @property
    def all_edges(self) -> list[tuple[int, int]]:
        """A list of all edges in the graph."""
        return [(i, j) for i in self.all_vertices for j in self.all_vertices if self.adjacency_matrix[i - 1, j - 1] > 0]

    @property
    def non_edges(self) -> list[tuple[int, int]]:
        """A list of all pairs `(i, j)` that are not edges in the graph."""
        return [
            (i + 1, j + 1)
            for i in range(self.n_vertices)
            for j in range(self.n_vertices)
            if self.adjacency_matrix[i, j] <= 0
        ]

    def __init__(self, n_vertices: int, edges: list[Edge]) -> None:
        """Initialises a Graph object.

        Args:
            n_vertices (int): The number of vertices in the graph.
            edges (list[Edge]): A list of edges in the graph.
        """
        self.n_vertices = n_vertices
        self.adjacency_matrix = np.zeros((n_vertices, n_vertices))
        for edge in edges:
            if len(edge) == 2:
                (from_vertex, to_vertex, weight) = (edge[0], edge[1], 1.0)
            else:
                (from_vertex, to_vertex, weight) = edge
            self.adjacency_matrix[from_vertex - 1, to_vertex - 1] = weight if weight != -1 else 0

    @staticmethod
    def read(file: TextIOWrapper) -> Graph:
        """Reads a graph from a file.

        Args:
            file (TextIOWrapper): The file to read the graph from.

        Returns:
            Graph: The graph read from the file.
        """
        m = np.loadtxt(file)
        g = Graph(m.shape[0], [])
        g.adjacency_matrix = m
        return g

    def store(self, file: TextIOWrapper) -> None:
        """Stores the graph in a file.

        Args:
            file (TextIOWrapper): The file to store the graph in.
        """
        np.savetxt(file, self.adjacency_matrix)

    @staticmethod
    def from_adjacency_matrix(
        adjacency_matrix: npt.NDArray[np.int_ | np.float64] | list[list[int]] | list[list[float]],
    ) -> Graph:
        """Creates a graph from an adjacency matrix.

        Args:
            adjacency_matrix (npt.NDArray[np.int_ | np.float64]): The adjacency matrix to create the graph from.

        Returns:
            Graph: The graph created from the adjacency matrix.
        """
        if isinstance(adjacency_matrix, list):
            adjacency_matrix = np.array(adjacency_matrix)
        g = Graph(adjacency_matrix.shape[0], [])
        g.adjacency_matrix = adjacency_matrix
        return g

    @staticmethod
    def deserialize(encoding: str) -> Graph:
        """Deserializes a graph from a string.

        Args:
            encoding (str): The string to deserialize the graph from.

        Returns:
            Graph: The deserialized graph.
        """
        m = np.array([[float(cell) for cell in line.split() if cell] for line in encoding.splitlines() if line.strip()])
        g = Graph(m.shape[0], [])
        g.adjacency_matrix = m
        return g

    def serialize(self) -> str:
        """Serializes the graph into a string.

        Returns:
            str: The serialized graph as a string.
        """
        return str(self.adjacency_matrix).replace("]", "").replace("[", "").replace("\n ", "\n")

    def plot(self) -> None:
        """Draws the graph using matplotlib and networkx."""
        g: nx.Graph = nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(g, seed=20)
        ax = plt.gca()
        nx.draw(
            g,
            pos,
            ax,
            arrows=True,
            with_labels=True,
            labels={i: i + 1 for i in range(self.n_vertices)},
        )
        nx.draw_networkx_edge_labels(
            g,
            pos,
            font_size=7,
            edge_labels={e: self.adjacency_matrix[int(e[0]), int(e[1])] for e in g.edges},
        )
        ax.set_axis_off()
        plt.show()

    def __eq__(self, value: object) -> bool:
        """Checks if two graphs are equal.

        Args:
            value (object): The other graph to compare to.

        Returns:
            bool: True if the graphs are equal, False otherwise.
        """
        if not isinstance(value, Graph):
            return False
        return cast(bool, np.array_equal(self.adjacency_matrix, value.adjacency_matrix))

    def __hash__(self) -> int:
        """Returns the hash of the graph.

        Returns:
            int: The hash of the graph.
        """
        return hash(self.adjacency_matrix)
