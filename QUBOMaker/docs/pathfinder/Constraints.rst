Constraints
===========

The *Pathfinder* submodule supports a total of 13 constraints. These constraints are translated
into their QUBO formulation automatically during the QUBO generation process, based on the selected encoding.

The following lists all supported constraints together with their properties and their QUBO formulations.
In the formula representation, :math:`\delta(x, v, i, \pi^{(i)})` is a function that returns 1 if vertex :math:`v` is at position :math:`i` of path :math:`\pi^{(i)}` and 0 otherwise.
The specific function depends on the selected :doc:`encoding <Encodings>`.

In each formula, :math:`x` is the binary variable assignment, :math:`N` is the maximum path length, :math:`V` is the set of vertices, :math:`E` is the set of edges, and :math:`A_{uv}` is the adjacency matrix of the graph.

PathIsValid
-----------
Enforces that a path is valid. A path is valid if:

#. For any two consecutive vertices, there exists an edge between them.
#. If the used encoding is :code:`ONE_HOT`, no two or more vertices are selected as the same position of the same path.
#. If the used encoding is :code:`DOMAIN_WALL`, the bitstring indicating the vertex occupying position :math:`j` of path :math:`\pi^{(i)}` is of the form :math:`111..10..0`, i.e. after the first 0, no more bits have the value 1.

Properties:
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\sum_{(u \\rightarrow v) \\not \\in E} \\sum_{i = 1}^{N}\\delta(x, \\pi, u, i)\\delta(x, \\pi, v, i+1)$$

*Encoding-specific penalties are further added to this expression if necessary.*

PathPositionIs
--------------
Enforces that a given position of a provided path is occupied by one of a set of vertices.

Properties:
    - :code:`position: int`: The position of the path subject to this constraint.
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that can occupy the position.
    - :code:`path: int`: The ID of the path this constraint should be applied to.

$$1 - \\sum_{v \\in V'} \\delta(x, \\pi, v, i)$$

PathStartsAt
-------------
Enforces that a given path starts at one of a given set of vertices.

Properties:
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that the path can start at.
    - :code:`path: int`: The ID of the path this constraint should be applied to.

$$1 - \\sum_{v \\in V'} \\delta(x, \\pi, v, 1)$$

PathEndsAt
-----------
Enforces that a given path ends at one of a given set of vertices.

Properties:
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that the path can end at.
    - :code:`path: int`: The ID of the path this constraint should be applied to.

$$\\sum_{i=2}^N \\left[ \\left(1 - \\sum_{v \\in V} \\delta(x, \\pi, v, i) \\right)^2 \\left(\\sum_{v \\not \\in V'} \\delta(x, \\pi, v, i - 1) \\right) \\right] + \\sum_{v \\not \\in V'} \\delta(x, \\pi, v, N)$$

PathContainsVerticesExactlyOnce
-------------------------------
Enforces that a given set of paths each contain each given vertex exactly once.

Properties:
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that should be contained exactly once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\left(1 - \\sum_{i = 1}^N \\delta(x, \\pi, v, i) \\right) ^2$$

PathContainsVerticesAtLeastOnce
-------------------------------
Enforces that a given set of paths each contain each given vertex at least once.

Properties:
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that should be contained at least once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\prod_{i=1}^N(1 - \\delta(x, v, i))$$

PathContainsVerticesAtMostOnce
-------------------------------
Enforces that a given set of paths each contain each given vertex at most once.

Properties:
    - :code:`vertex_ids: list[int]`: A list of IDs of the vertices that should be contained at most once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\sum_{1 \\leq i \\lt j \\lt N}\\delta(x, v,i)\\delta(x, v,j)$$

PathContainsEdgesExactlyOnce
-------------------------------
Enforces that a given set of paths each contain each given edge exactly once.

Properties:
    - :code:`edges: list[tuple[int, int]]`: A list of the edges that should be contained exactly once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\left( 1 - \\sum_{i=1}^{N}\\delta(x, \\pi, u, i)\\delta(x, \\pi, v, i + 1) \\right)^2$$

PathContainsEdgesAtLeastOnce
-------------------------------
Enforces that a given set of paths each contain each given edge at least once.

Properties:
    - :code:`edges: list[tuple[int, int]]`: A list of the edges that should be contained at least once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\prod_{i=1}^N(1 - \\delta(x, \\pi, u, i)\\delta(x, \\pi, v, i +1))$$

PathContainsEdgesAtMostOnce
-------------------------------
Enforces that a given set of paths each contain each given edge at most once.

Properties:
    - :code:`edges: list[tuple[int, int]]`: A list of the edges that should be contained at most once in each path.
    - :code:`path_ids: list[int]`: A list of IDs of the paths this constraint should be applied to.

$$\\sum_{1 \\leq i \\lt j \\leq N}(\\delta(x, \\pi, u, i)\\delta(x, \\pi, v,i+1))(\\delta(x, \\pi, u,j)\\delta(x, \\pi, v,j+1))$$

PrecedenceConstraint
--------------------
For the given vertices :math:`u` and :math:`v`, enforces that :math:`u` is visited before :math:`v` in the path.

Properties:
    - :code:`pre: int`: The ID of the preceding vertex.
    - :code:`post: int`: The ID of the preceded vertex.

$$\\sum_{i=1}^N\\delta(x, \\pi, v,i)\\prod_{j=1}^{i-1}(1-\\delta(x, \\pi, u,j))$$

PathsShareNoVertices
--------------------
Enforces that two given paths share no vertices.

Properties
    - :code:`path_one: int`: The ID of the first path.
    - :code:`path_two: int`: The ID of the second path.

$$\\sum_{v \\in V} \\left[ \\left(\\sum_{i=1}^N \\delta(x, \\pi^{(1)}, v, i) \\right) \\left(\\sum_{i=1}^N \\delta(x, \\pi^{(2)}, v, i) \\right) \\right]$$

PathsShareNoEdges
-----------------
Enforces that two given paths share no edges.

Properties
    - :code:`path_one: int`: The ID of the first path.
    - :code:`path_two: int`: The ID of the second path.

$$\\sum_{(u \\rightarrow v) \\in E} \\left[ \\left( \\sum_{i=1}^{N} \\delta(x, \\pi^{(1)}, u, i) \\delta(x, \\pi^{(1)}, v, i + 1) \\right) \\left(\\sum_{i=1}^{N} \\delta(x, \\pi^{(2)}, u, i) \\delta(x, \\pi^{(2)}, v, i + 1) \\right) \\right]$$

MinimizePathLength
------------------
Enforces that the length of a given path is minimized.

Properties
    - :code:`path_ids: int`: The ID of the paths this constraint should be applied to.

$$\\sum_{(u \\rightarrow v) \\in E} \\sum_{i = 1}^{N} A_{uv}\\delta(x, \\pi, u, i)\\delta(x, \\pi, v, i+1)$$

MaximizePathLength
------------------
Enforces that the length of a given path is maximized.

Properties
    - :code:`path_ids: int`: The ID of the paths this constraint should be applied to.

$$-\\sum_{(u \\rightarrow v) \\in E} \\sum_{i = 1}^{N} A_{uv}\\delta(x, \\pi, u, i)\\delta(x, \\pi, v, i+1)$$
