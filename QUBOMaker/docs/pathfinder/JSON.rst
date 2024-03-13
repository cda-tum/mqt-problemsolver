JSON Encoding
=============

Instead of using the programming interface for creating a QUBOGenerator, the *Pathfinder* submodule also supports a JSON format.

With a given input file in JSON format, the QUBOGenerator can be created using the following code:

.. code-block:: python

    with Path.open("input.json") as file:
        generator_new = pf.PathFindingQUBOGenerator.from_json(file.read(), graph)

The JSON input contains the definitions of problem constraints to be used for the QUBO formulation, as well as general settings such as the desired encoding choice.

The format for the JSON input is defined as:

.. code-block::

    {
        "settings": {
            "encoding": one of ["ONE_HOT", "UNARY", "DOMAIN_WALL", "BINARY"],
            "n_paths": integer,
            "max_path_length": integer,
            "loops": boolean,
        },
        "objective_function": Constraint,
        "constraints": array[Constraint]
    }

Individual constraints are defined based on their JSON definitions provided `here <https://github.com/cda-tum/mqt-qubomaker/tree/main/src/mqt/qubomaker/pathfinder/resources/constraints>`_.
