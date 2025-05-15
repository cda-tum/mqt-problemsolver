Solution Class
==============

It is a class that is used to store all the information about the solution of the problem.

Solution Attributes
-------------------

The class has the following attributes:

- energies: list[float], i.e. the list of energies of the solution obtained in each run
- best_energy, i.e. the lowest energy obtained in all the runs
- best_solution: dict[str, Any], i.e. the best solution obtained in all the runs in binary variables
- best_solution_original_var: dict[str, Any], i.e. the best solution obtained in all the runs in variables originally declared
- solutions_original_var: list[dict[str, Any]], i.e. the list of solutions obtained in each run in variables originally declared
- solutions: list[dict[str, Any]], i.e. the list of solutions obtained in each run in binary variables
- time: float: the time taken to solve the problem
- solver_info: dict[str, Any], i.e. the information about the solver setting to solve the problem


Solution Methods
----------------

The class has the following methods for helping users in analyzing the solution:

- *optimal_solution_cost_functions_values()*: This method is used to get the cost functions values of the best solution obtained in all the runs
- *check_constraint_optimal_solution()*: This method is used to check if the best solution obtained in all the runs satisfies the constraints
- *check_constraint_all_solutions()*: This method is used to check if all the solutions obtained in all the runs satisfy the constraints
- *show_cumulative(save: bool = False, show: bool = True, filename: str = "", label: str = "", latex: bool = False)* : This method is used to show the cumulative plot of the energies obtained in all the runs. The parameters are:
    - save: bool, i.e. whether to save the plot or not in a file
    - show: bool, i.e. whether to show the plot or not
    - filename: str, i.e. the name of the file where the plot will be saved
    - label: str, i.e. the label of the plot
    - latex: bool, i.e. whether to show the plot in latex format or not
- *valid_solutions(weak: bool = True)*: This method is used to get the rate of valid solutions obtained in all the runs. The parameters are:
    - weak: bool, i.e. whether to consider in the evaluation the weak constraints
- *p_range(ref_value: float | None = None)*: This method is used to get the p-range of the best solution obtained in all the runs, which is the probability of obtaining a final energy lower than a certain value. The parameters are:
    - ref_value: float | None, i.e. the reference value to calculate the p-value
- *tts(ref_value: float | None = None, target_probability: float = 0.99)*: This method is used to get the time-to-solution of the best solution obtained in all the runs, which is the time required to obtain a solution with a certain probability. The parameters are:
    - ref_value: float | None, i.e. the reference value to calculate the p-range
    - target_probability: float, i.e. the target probability to calculate the time-to-solution
- *wring_json_reports(filename: str = "report", weak: bool = False, ref_value: float | None = None, target_probability: float = 0.99, problem_features: bool = False)* : This method is used to write the reports in json format. The parameters are:
    - filename: str, i.e. the name of the file where the report will be saved
    - weak: bool, i.e. whether to consider in the evaluation the weak constraints
    - ref_value: float | None, i.e. the reference value to calculate the p-range
    - target_probability: float, i.e. the target probability to calculate the time-to-solution
    - problem_features: bool, i.e. whether to show the problem features in the report or not



Examples:
---------

.. code-block:: python

    from mqt.qao.constraints import Constraints
    from mqt.qao.variables import Variables
    from mqt.qao.objectivefunction import ObjectiveFunction
    from mqt.qao.problem import Problem
    from mqt.qao.solver import Solver

    variables = Variables()
    m1 = variables.add_continuous_variables_array(
        "M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2"
    )
    m2 = variables.add_continuous_variables_array(
        "M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2"
    )
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(np.matmul(m1, m2).item(0, 0))
    constraint = Constraints()
    constraint.add_constraint(constraint_expr, variable_precision=True)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_simulated_annealing(
        problem,
        max_lambda_update=max_lambda_update,
        lambda_update_mechanism=lambda_update,
        lambda_strategy=lambda_strategy,
    )

    print(solution.optimal_solution_cost_functions_values())
    print(solution.check_constraint_optimal_solution())
    print(solution.check_constraint_all_solutions())
    solution.show_cumulative()
    print(solution.valid_solutions())
    print(solution.p_range())
    print(solution.tts())
    solution.wring_json_reports()
