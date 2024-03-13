TSPLib Input
============

`TSPLib <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>`_ provides a library of sample
instances for different pathfinding problems, as well as a format for defining them.

This module provides ways of reading TSPLib input files using the `tsplib95 python library <https://tsplib95.readthedocs.io/en/stable/>`_ and converting them into a QUBOGenerator with
the corresponding constraints.

.. code-block:: python

    from mqt.qubomaker.pathfinder import EncodingType, from_tsplib_problem
    import tsplib95

    problem = tsplib95.load("tsp_lib_file.tsp")

    generator = from_tsplib_problem(problem, EncodingType.ONE_HOT)
