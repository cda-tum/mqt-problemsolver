Problem Class
=============

It may be useful to have a class that represents a problem. It includes the variables, constraints and objective function. This class can be used for constructing the quantum-compliant cost function.

Problem declarations
--------------------

The class provides a  method for declaring the problem:

- *create_problem(var: Variables, constraint: Constraints, objective_functions: ObjectiveFunction)* : This method is used to declare the problem.
    - *variables* instances of the Variables class
    - *constraints* instances of the Constraints class
    - *objective_functions* instances of the ObjectiveFunction class.

and a method for obtaining the HUBO or PUBO formulation of the problem as qubovert PUBO object:

- *write_the_final_cost_function( lambda_strategy: str, lambda_value: float = 1)* : This method is used to obtain the HUBO or PUBO formulation of the problem as qubovert PUBO object. The method takes two arguments:
    - *lambda_strategy* The strategy to be used for the conversion of the problem to HUBO or PUBO. The possible values are:
        - upper_bound_only_positive
        - maximum_coefficient
        - VLM
        - MOC
        - MOMC
        - upper lower bound naive
        - upper lower bound posiform and negaform method
        - manual
    - *lambda_value*  The value of the lambda parameter if the use want to manually select it. The default value is 1.0.

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
    constraint.add_constraint("c >= 1", True, True, False)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(a0 + b0 * c0 + c0**2)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    pubo = problem.write_the_final_cost_function(lambda_strategy)
