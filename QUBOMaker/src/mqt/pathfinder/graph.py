from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from io import TextIOWrapper

# Python 3.12:
# type Edge = tuple[int, int] | tuple[int, int, int] | tuple[int, int, float]
Edge = tuple[int, int] | tuple[int, int, int] | tuple[int, int, float]


class Graph:
    n_vertices: int
    adjacency_matrix: npt.NDArray[np.int_ | np.float64]

    @property
    def all_vertices(self) -> list[int]:
        return list(range(1, self.n_vertices + 1))

    def __init__(self, n_vertices: int, edges: list[Edge]) -> None:
        self.n_vertices = n_vertices
        self.adjacency_matrix = np.ones((n_vertices, n_vertices)) * -1
        for edge in edges:
            if len(edge) == 2:
                (from_vertex, to_vertex, weight) = (edge[0], edge[1], 1.0)
            else:
                (from_vertex, to_vertex, weight) = edge
            self.adjacency_matrix[from_vertex - 1, to_vertex - 1] = weight

    @staticmethod
    def read(file: TextIOWrapper) -> Graph:
        m = np.mat(np.loadtxt(file))
        g = Graph(m.shape[0], [])
        g.adjacency_matrix = m
        return g

    def store(self, file: TextIOWrapper) -> None:
        np.savetxt(file, self.adjacency_matrix)

    @staticmethod
    def deserialize(encoding: str) -> Graph:
        lines = [x.strip() for x in encoding.strip().split("\n") if x.strip()]
        values = [[float(cell.strip()) for cell in line.split(" ") if cell.strip()] for line in lines]
        m = np.mat(values)
        g = Graph(m.shape[0], [])
        g.adjacency_matrix = m
        return g

    def serialize(self) -> str:
        return str(self.adjacency_matrix).replace("]", "").replace("[", "").replace("\n ", "\n")

    def plot(self) -> None:
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
