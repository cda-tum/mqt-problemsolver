Graphical User Interface
========================

A web-based GUI for the Pathfinder submodule can be accessed through `GitHub pages <https://cda-tum.github.io/mqt-qubomaker/>`_.
The GUI is a simple interface that allows users to define their problem graph through its adjacency matrix,
and then select a set of constraints to define a problem instance.

The GUI supports all main constraint types implemented for the *Pathfinder* submodule. However, specific
constraint-specific settings (such as applying "*path contains*"-constraints just to single paths) are not always supported.

Once the constraints have been chosen and a desired encoding is selected, the "generate" button can be used to generate a JSON representation of
the problem instance. This can then be used as input for the *Pathfinder* submodule, as defined in the :doc:`JSON documentation <JSON>`.
