ObjectiveFunction Class
=======================

It manages the objective functions of the optimization problem. it permits to specify the weights of the objectives in case of multi-objective optimization and the optimization directions. Since the variables can be declared in array form, matrix expressions for the objective functions are supported.

Objective functions declarations
--------------------------------

The class provides methods to declare variables:

- *add_objective_function(objective_function: Expr, minimization: bool = True, weight: float = 1)* : add an objective function to the optimization problem.
    - *objective_function* is an expression of the variables of the optimization problem.
    - *minimization* parameter specifies if the objective function is to be minimized or maximized (optimization direction).
    - *weight* parameter is the weight of the objective function in case of multi-objective optimization.

Example:
--------

.. code-block:: python

    from mqt.qao.constraints import Constraints
    from mqt.qao.variables import Variables
    from mqt.qao.objectivefunction import ObjectiveFunction

    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25, "", "")
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(a0 + b0 * c0 + c0**2)
