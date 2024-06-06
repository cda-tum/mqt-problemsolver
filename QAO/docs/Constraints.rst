Constraints Class
=================

It manages the constraints of the problem. It is used to store the constraints and to check if a solution satisfies them.

Constraints types supported
---------------------------

Types of constraints supported:

- *Equality*, writing the equality as a quadratic penalty function, i.e. sum = b imposed with g = (sum-b)^2
- *Inequality*, moving to equality with the continuous auxiliary variables a to be expanded with the encoding technique of the Variables class
- *Not constraint* among binary variables, i.e. Not(a) = b imposed with g = 2ab - a - b + 1
- *And constraint* among binary variables, i.e.  a and b = c imposed with  ab -2(a+b)c + 3c
- *Or constraint* among binary variables, i.e. a or b = c imposed with ab + (a+b)(1-2c) + c
- *Xor constraint* among binary variables, i.e. a xor b = c imposed with 2ab - 2(a+b)c - 4(a+b)\_aux+4_aux c +a+b+c+4+\_aux

Constraints declarations
------------------------

The class provides methods to declare variables:

- *add_constraint(expression: str, hard: bool = True, variable_precision: bool = True)*: adds a constraint to the list of constraints.
    - *expression* is a string that represents the constraint
    - *hard* parameter is a boolean that indicates if the constraint is hard or soft.
    - *variable_precision* parameter is a boolean that indicates if the constraint is to be considered in the precision of the variables.

Example:
--------

.. code-block:: python

    from mqt.qao.constraints import Constraints
    from mqt.qao.variables import Variables

    constraint = Constraints()
    variables = Variables()
    variables.add_binary_variable("a")
    variables.add_binary_variable("b")
    variables.add_binary_variable("c")
    variables.add_discrete_variable("d", [-1, 1, 3])
    variables.add_continuous_variable("e", -2, 2, 0.25, "", "")
    constraint.add_constraint("~a = b", True, True, False)
    constraint.add_constraint("a | b = c", True, True, False)
    constraint.add_constraint("d + e <= 1", True, True, False)
