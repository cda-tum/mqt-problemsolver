.. MQT QUBOMaker documentation master file, created by
   sphinx-quickstart on Wed Mar 13 10:46:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MQT QUBOMaker's documentation!
=========================================

MQT QUBOMaker is a framework for creating QUBO formulations for diverse optimization problems. These formulations can be used with a wide range of quantum algorithms to find approximate solutions to the problems.

The framework is designed to be user-friendly and to provide a high-level interface for creating QUBO formulations, not requiring any background knowledge in quantum computing or QUBO problems to solve domain-specific problems. It is also designed to be extensible, so that new optimization problems can be added to the framework with relative ease.

MQT QUBOMaker was developed as part of the `Munich Quantum Toolkit <https://mqt.readthedocs.io>`_ (*MQT*) by the `Chair for Design Automation <https://www.cda.cit.tum.de/>`_ at the `Technical University of Munich <https://www.tum.de>`_.


Currently, MQT QUBOMaker supports the following optimization problem types:
   - `Pathfinding Problems <pathfinder/Pathfinder.html>`_

In addition to a semantic selection of problem constraints, MQT QUBOMaker also provides a `graphical user interface <pathfinder/GUI.html>`_ for the Pathfinding submodule, which allows users to interactively define pathfinding problems to be passed to the framework.

MQT QUBOMaker supports several encoding schemes, allowing end users to easily compare and evaluate different encodings without the need for manual rewriting of the problems.

If you are interested in the theory behind MQT QUBOMaker, have a look at the publications in the :doc:`references list <References>`.

---

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :glob:

   Quickstart
   Installation
   QUBOGenerator
   pathfinder/index
   References

.. toctree::
   :maxdepth: 1
   :caption: Developers
   :glob:

   Contributing
   DevelopmentGuide
   Support
