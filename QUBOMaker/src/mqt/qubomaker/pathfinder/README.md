This submodule of MQT QUBOMaker is responsible for the QUBO formulation of pathfinding problems.

### Settings

Constructing QUBO formulation for pathfinding problems requires a `PathfindingQUBOGenerator` instance with the corresponding settings:

- `encoding_type`: An element of the `EncodingTypes` enum that represents the variable encoding scheme to be used. One of `ONE_HOT`, `DOMAIN_WALL`, or `BINARY`.
- `n_paths`: The number of paths to be found. For most problem instances, this value will be `1`.
- `max_path_length`: The maximum number of vertices a found path can consist of. Required to determine the number of binary variables that have to be used for the QUBO formulation.
- `loops`: Determines, whether the path represents a loop, i.e., the final vertex in its sequence is connected back to the starting vertex.

An example settings instance can be constructed as follows:

```python3
import mqt.qubomaker.pathfinder as pf

settings = pf.PathFindingQUBOGeneratorSettings(
    encoding_type=pf.EncodingType.ONE_HOT, n_paths=1, max_path_length=4, loops=True
)
```

### PathFindingQUBOGenerator

The `PathFindingQUBOGenerator` class represents the main QUBO factory for pathfinding problems. It can be set up with predefined settings and populated with problem-specific constraints.

To create a `PathFindingQUBOGenerator`, a `Graph` instance further has to be provided, representing the graph to be investigated for the problem instance.

```python3
...

qubo_generator = pf.PathFindingQUBOGenerator(
    objective_function=None, graph=my_graph, settings=settings
)
```

When creating a `PathFindingQUBOGenerator` instance, an objective function, as discussed below, can be added to add an optimization criterion.

### Cost Functions

The `pathfinder` module provides cost functions representing various constraints related to pathfinding problems. The following constraints are supported:

- `PathIsValid`: Checks, whether the encoding represents a valid path. Should be included in most cases.
- `PathPositionIs`: Enforces that one of a set of vertices is located at a given position of a graph.
- `PathStartsAt`: Enforces that one of a set of vertices is located at the start of a graph.
- `PathEndsAt`: Enforces that one of a set of vertices is located the the end of a graph.
- `PathContainsVerticesExactlyOnce`: Enforces that each element of a given set of vertices appears exactly once in a path.
- `PathContainsVerticesAtLeastOnce`: Enforces that each element of a given set of vertices appears at least once in a path.
- `PathContainsVerticesAtMostOnce`: Enforces that each element of a given set of vertices appears at most once in a path.
- `PathContainsEdgesExactlyOnce`: Enforces that each element of a given set of edges appears exactly once in a path.
- `PathContainsEdgesAtLeastOnce`: Enforces that each element of a given set of edges appears at least once in a path.
- `PathContainsEdgesAtMostOnce`: Enforces that each element of a given set of edges appears at most once in a path.
- `PrecedenceConstraint`: Enforces, that a given vertex does not appear in a path that didn't visit another given vertex first.
- `PathsShareNoVertices`: Enforces, that two provided paths do not share any vertices.
- `PathsShareNoEdges`: Enforces, that two provided paths do not share any edges.

Furthermore, the `pathfinder` module also provides two objective functions:

- `MinimizePathLength`: Adds a cost to the total cost function that penalizes paths with a higher total weight.
- `MaximizePathLength`: Adds a cost to the total cost function that penalizes paths with a lower total weight.

These constraints are represented by classes, and instances of the classes can be created to define the specific constraint and added to the QUBOGenerator.

```python3
...

contains_all_vertices = pf.PathContainsVerticesExactlyOnce(
    vertex_ids=graph.all_vertices, path_ids=[1]
)
starts_at_1 = pf.PathStartsAt(vertex_ids=[1], path=1)

qubo_generator.add_constraint(contains_all_vertices)
qubo_generator.add_constraint(starts_at_1)
```
