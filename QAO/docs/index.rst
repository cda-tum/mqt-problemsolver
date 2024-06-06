Welcome to MQT Quantum Auto Optimizer's documentation!
======================================================

MQT Quantum Auto Optimizer is a tool for automatic framework to assist users in utilizing quantum solvers for optimization tasks while preserving interfaces that closely resemble conventional optimization practices and is developed as part of the `Munich Quantum Toolkit <https://mqt.readthedocs.io>`_ (*MQT*) by the `Chair for Design Automation <https://www.cda.cit.tum.de/>`_ at the `Technical University of Munich <https://www.tum.de>`_.

From a user's perspective, the framework is used as follows:

.. image:: /_static/mqt_qao.png
   :width: 100%
   :alt: Illustration of the MQT Quantum Auto Optimizer framework
   :align: center

The framework is designed to be user-friendly and to provide a high-level interface for assisting assist users in utilizing quantum solvers for optimization tasks, not requiring any prior knowledge of quantum computing.
The framework prompts users to specify variables, optimization criteria, as well as validity constraints and, afterwards, allows them to choose the desired solver. Subsequently, it automatically transforms the problem description into a format compatible with the chosen solver and provides the resulting solution. Additionally, the framework offers instruments for analyzing solution validity and quality.

If you are interested in the theory behind MQT Quantum Auto Optimizer, have a look at the publications in the :doc:`references list <References>`.


----

 .. toctree::
    :hidden:

    self

 .. toctree::
    :maxdepth: 1
    :caption: User Guide
    :glob:

    Quickstart
    Usage
    Variable
    Constraints
    ObjectiveFunction
    Problem
    Solver
    Solution
    References

 .. toctree::
    :maxdepth: 1
    :caption: Developers
    :glob:

    Contributing
    DevelopmentGuide
    Support
