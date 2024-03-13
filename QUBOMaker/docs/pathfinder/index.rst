Pathfinder Submodule
====================

This module implements MQT QUBOMaker for pathfinding problems on directed and undirected graphs in the form of the :code:`PathFindingQUBOGenerator` class, a specialization of the general :code:`QUBOGenerator` class.

It supports a set of various :doc:`constraints <Constraints>` that can be used to model a variety of different pathfinding problems.

In addition to that, it also provides three :doc:`encoding schemes <Encodings>` that can be selected for the construction of QUBO formulations.

Finally, the :doc:`GUI <GUI>` provides a graphical user interface for the module, which can be used to interactively define pathfinding problems.

In addition to that, the submodule accepts several input formats for the problem instance. A :doc:`JSON format <JSON>` can be used to define problem constraints and settings. Furthermore, the established :doc:`TSPLib format<TSPLib>` can be passed to the framework directly, generating the required constraints from the problem instance.

The :code:`PathFindingQUBOGenerator` class can be instantiated like this:

.. code-block:: python

   import mqt.qubomaker.pathfinder as pf

   ...

   generator = pf.PathFindingQUBOGenerator(
       objective_function=pf.MinimizePathLength(path_ids=[1]),
       graph=graph,
       settings=settings,
   )

Here, the :code:`objective_function` parameter can represent any objective function for the optimization procedure (:code:`MinimizePathLength` or :code:`MaximizePathLength`). The :code:`graph` parameter is the graph on which the problem is defined. Finally the :code:`settings` parameter is a :code:`PathFindingQUBOGeneratorSettings` object that defines settings for the QUBO generator:

- :code:`encoding_type`: The encoding scheme to use for the QUBO formulation.
- :code:`n_paths`: The number of paths to be searched for.
- :code:`max_path_length`: The maximum length of a path.
- :code:`loops`: A boolean value indicating, whether the found paths should be interpreted as loops.

An example settings definition may look like:

.. code-block:: python

   import mqt.qubomaker.pathfinder as pf

   settings = pf.PathFindingQUBOGeneratorSettings(
       encoding_type=pf.EncodingType.BINARY,
       n_paths=1,
       max_path_length=4,
       loops=False,
   )


.. toctree::
   :maxdepth: 1
   :caption: Pathfinder Features

   Constraints
   Encodings
   GUI
   JSON
   TSPLib
