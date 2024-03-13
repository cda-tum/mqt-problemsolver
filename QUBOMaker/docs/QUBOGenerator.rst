QUBOGenerator
================

The :code:`QUBOGenerator` class is the main concept of the package. It provides methods for the
automatic construction of QUBO formulations, including the computation of penalty factors for individual
cost functions, the creation of auxiliary variables to reduce cost functions to quadratic order, and the
translation into different output formats.

A :code:`QUBOGenerator` object represents a single problem instance and collects constraints and cost functions related to it.
It is an abstract base class for specialized QUBO generators, such as the :code:`PathFindingQUBOGenerator` to extend.

It provides the following output formats:

Simplified QUBO formula
------------------------

The QUBO formulation of the corresponding problem instance as a simplified formula, including sum expressions and additional functions
if necessary, to provide a human-readable representation of the QUBO.

Expanded QUBO formula
---------------------

The QUBO formulation of the corresponding problem expanded as a polynomial expression, without any sums or additional functions.
This expression also includes additional auxiliary variables, if necessary, to reduce the order of the products to quadratic expressions.

QUBO matrix
-----------

The QUBO as a triangular matrix :math:`Q`, such that the QUBO problem can be represented as

$$\\mathbf{x}* = \\text{argmin}_{\\mathbf{x}} \\mathbf{x}^T Q \\mathbf{x}$$

Entry :math:`Q_{ij}` of the matrix represents the coefficient of the quadratic term :math:`x_i * x_j` in the expanded QUBO expression.

Quantum Operator
----------------

The QUBO as a hamiltonian operator whose minimum eigenstate encodes to the optimal assignment.

Quantum algorithms such as Grover adaptive search, QAOA, VQE, or QPE can be used to solve the eigenvalue problem.

Quantum Circuit
---------------

Qiskit's QAOA implementation to solve the given QUBO problem. Returns an instance of the qiskit :code:`QAOA` class that can be
used to optimize the problem's hamiltonian operator. This allows end users to run the quantum algorithms without needing to interact with any
quantum computing concepts directly themselves.
