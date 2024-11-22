KarpGraphs Class
================

Overview
--------
The **karp** module provides tools to solve and validate various NP-complete problems defined by Karp. The module is subdivided into three classes: **KarpGraphs**, **KarpSets**, and **KarpNumber**.

This document focuses on the KarpGraph class, which contains methods for solving graph-based problems, such as **graph coloring**, **vertex cover**, and **clique**. Among these, the ``graph_coloring`` method is detailed below.

Class: ``KarpGraphs``
---------------------

Method: ``graph_coloring``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method sets up and optionally solves the graph coloring problem for a given input graph and number of colors.

**Arguments**

- **``input_data``** *(str | nx.Graph)*: Input graph provided as a file path or a NetworkX graph.
- **``num_colors``** *(int)*: The number of colors to use for coloring the graph.
- **``a``** *(float)*: Penalty parameter for constraint violations (default = 1).
- **``solve``** *(bool)*: If ``True``, solves the problem; if ``False``, returns the problem setup (default = ``False``).
- **``solver_method``** *(Callable | None)*: Optional solver function. If ``None``, uses a default solver.
- **``read_solution``** *(Literal["print", "file"] | None)*: Output format for the solution, either printed or saved to a file.
- **``solver_params``** *(dict | None)*: Additional parameters for the solver.
- **``txt_outputname``** *(str | None)*: File name for saving the solution.
- **``visualize``** *(bool)*: If ``True``, visualizes the solution graph (default = ``False``).

**Returns**

- If ``solve`` is ``False``: Returns a ``Problem`` object representing the graph coloring problem.
- If ``solve`` is ``True``: Returns a list of tuples, each representing a vertex and its assigned color.

**Example Usage**

.. code-block:: python

    from karp import KarpGraphs
    import networkx as nx

    # Define a simple graph
    G = nx.cycle_graph(4)

    # Solve graph coloring for 2 colors
    solution = KarpGraphs.graph_coloring(
        input_data=G, num_colors=2, solve=True, visualize=True
    )

    print(solution)

Visualization
~~~~~~~~~~~~~
If the ``visualize`` option is set to ``True``, the graph is displayed with nodes colored according to the solution. Nodes with the same color indicate a valid coloring where no adjacent nodes share a color.

**Example Visualization:**

The visualization shows the graph structure with each node assigned a unique color. This provides intuitive feedback on the solution's correctness.

Solution Validation
~~~~~~~~~~~~~~~~~~~
The method checks the validity of the solution using the following criteria:

- **Missing Nodes**: Ensures all nodes are assigned a color.
- **Conflicting Edges**: Ensures no two adjacent nodes share the same color.

Error Handling
~~~~~~~~~~~~~~
If a valid solution cannot be found, the method raises detailed validation errors, including:

- Nodes not assigned a color.
- Edges connecting nodes with the same color.

**Output Example**

For a graph with vertices ``[1, 2, 3, 4]`` and solution:

.. code-block:: text

    Vertex: 1 Color: 1
    Vertex: 2 Color: 2
    Vertex: 3 Color: 1
    Vertex: 4 Color: 2

The graph visualization highlights the node assignments with distinct colors.
