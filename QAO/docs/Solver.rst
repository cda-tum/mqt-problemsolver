Solver Class
============

It interface the problems with the quantum solvers.

Supported solvers
-----------------

The framework currently supports the following solvers:ù

- D-Wave quantum annealer
- D-Wave simulated annealer
- qiskit Quantum Approximate Optimization Algorithm (QAOA)
- qiskit Variational Quantum Eigensolver (VQE)
- qiskit Grover Adaptive Search (GAS)

Solver Selection and Configuration
----------------------------------

The class provides for exploiting the solver:

- solve_simulated_annealing(
        problem: Problem,
        auto_setting: bool = False,
        beta_range: list[float] | None = None,
        num_reads: int = 100,
        annealing_time: int = 100,
        num_sweeps_per_beta: int = 1,
        beta_schedule_type: str = "geometric",
        seed: int | None = None,
        initial_states: SampleSet | None = None,
        initial_states_generator: str = "random",
        max_lambda_update: int = 5,
        lambda_update_mechanism: str = "sequential penalty increase",
        lambda_strategy: str = "upper lower bound posiform and negaform method",
        lambda_value: float = 1.0,
        save_time: bool = False,
    ) : Solve the problem using the simulated annealer. The parameters are:
        - *problem*: the problem to solve
        - *auto_setting*: if True, the parameters are automatically set
        - *beta_range*: the range of beta values to use
        - *num_reads*: the number of reads
        - *annealing_time*: the annealing time
        - *num_sweeps_per_beta*: the number of sweeps per beta
        - *beta_schedule_type*: the beta schedule type
        - *seed*: the seed
        - *initial_states*: the initial states
        - *initial_states_generator*: the initial states generator
        - *max_lambda_update*: the maximum lambda update if the constraints are not satisfied
        - *lambda_update_mechanism*: the lambda update mechanism among:
            - *sequential penalty increase*
            - *scaled sequential penalty increase*
            - *binary search penalty algorithm*
- solve_dwave_quantum_annealer(
        problem: Problem,
        token: str,
        auto_setting: bool = False,
        failover: bool = True,
        config_file: str | None = None,
        endpoint: str | None = None,
        solver: dict[str, str] | str = "Advantage_system4.1",
        annealing_time_scheduling: float | list[list[float]] = 20.0,
        num_reads: int = 100,
        auto_scale: bool = True,
        flux_drift_compensation: bool = True,
        initial_state: dict[str, int] | None = None,
        programming_thermalization: float = 1000.0,
        readout_thermalization: float = 0.0,
        reduce_intersample_correlation: bool = True,
        max_lambda_update: int = 5,
        lambda_update_mechanism: str = "sequential penalty increase",
        lambda_strategy: str = "upper lower bound posiform and negaform method",
        lambda_value: float = 1.0,
        save_time: bool = False,
        save_compilation_time: bool = False,
    ) : Solve the problem using the D-Wave quantum annealer. The parameters are:
        - *problem*: the problem to solve
        - *token*: the token to access the D-Wave API
        - *auto_setting*: if True, the parameters are automatically set
        - *failover*: if True, the failover is enabled
        - *config_file*: the configuration file
        - *endpoint*: the endpoint
        - *solver*: the solver to use
        - *annealing_time_scheduling*: the annealing time scheduling
        - *num_reads*: the number of reads
        - *auto_scale*: if True, the problem is automatically scaled
        - *flux_drift_compensation*: if True, the flux drift compensation is enabled
        - *initial_state*: the initial state
        - *programming_thermalization*: the programming thermalization
        - *readout_thermalization*: the readout thermalization
        - *reduce_intersample_correlation*: if True, the intersample correlation is reduced
        - *max_lambda_update*: the maximum lambda update if the constraints are not satisfied
        - *lambda_update_mechanism*: the lambda update mechanism among:
            - *sequential penalty increase*
            - *scaled sequential penalty increase*
            - *binary search penalty algorithm*
- solve_grover_adaptive_search_qubo(
        problem: Problem,
        auto_setting: bool = False,
        qubit_values: int = 0,
        coeff_precision: float = 1.0,
        threshold: int = 10,
        num_runs: int = 10,
        max_lambda_update: int = 5,
        boundaries_estimation_method: str = "",
        lambda_update_mechanism: str = "sequential penalty increase",
        lambda_strategy: str = "upper lower bound posiform and negaform method",
        lambda_value: float = 1.0,
        save_time: bool = False,
        save_compilation_time: bool = False,
    ) : Solve the problem using the Grover Adaptive Search. The parameters are:
        - *problem*: the problem to solve
        - *auto_setting*: if True, the parameters are automatically set
        - *qubit_values*: the number of qubit values, if the user want to specify it manually
        - *coeff_precision*: the coefficient precision
        - *threshold*: the threshold
        - *num_runs*: the number of runs
        - *max_lambda_update*: the maximum lambda update if the constraints are not satisfied
        - *boundaries_estimation_method*: the boundaries estimation method for estimating the necessary number of qubit value
        - *lambda_update_mechanism*: the lambda update mechanism among:
            - *sequential penalty increase*
            - *scaled sequential penalty increase*
            - *binary search penalty algorithm*
- solve_qaoa_qubo(
        problem: Problem,
        auto_setting: bool = False,
        num_runs: int = 10,
        optimizer: Optimizer | None = None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer: QuantumCircuit = None,
        initial_point: np.ndarray[Any, Any] | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray[Any, Any], float, float], None] | None = None,
        max_lambda_update: int = 5,
        lambda_update_mechanism: str = "sequential penalty increase",
        lambda_strategy: str = "upper lower bound posiform and negaform method",
        lambda_value: float = 1.0,
        save_time: bool = False,
        save_compilation_time: bool = False,
    ) : Solve the problem using the Quantum Approximate Optimization Algorithm. The parameters are:
        - *problem*: the problem to solve
        - *auto_setting*: if True, the parameters are automatically set
        - *num_runs*: the number of runs
        - *optimizer*: the optimizer
        - *reps*: the number of repetitions
        - *initial_state*: the initial state
        - *mixer*: the mixer
        - *initial_point*: the initial point
        - *aggregation*: the aggregation function
        - *callback*: the callback function
        - *max_lambda_update*: the maximum lambda update if the constraints are not satisfied
        - *lambda_update_mechanism*: the lambda update mechanism among:
            - *sequential penalty increase*
            - *scaled sequential penalty increase*
            - *binary search penalty algorithm*
- solve_vqe_qubo(
        self,
        problem: Problem,
        auto_setting: bool = False,
        num_runs: int = 10,
        optimizer: Optimizer | None = None,
        ansatz: QuantumCircuit | None = None,
        initial_point: np.ndarray[Any, Any] | None = None,
        aggregation: float | Callable[[list[float]], float] | None = None,
        callback: Callable[[int, np.ndarray[Any, Any], float, float], None] | None = None,
        max_lambda_update: int = 5,
        lambda_update_mechanism: str = "sequential penalty increase",
        lambda_strategy: str = "upper lower bound posiform and negaform method",
        lambda_value: float = 1.0,
        save_time: bool = False,
        save_compilation_time: bool = False,
    ) : Solve the problem using the Variational Quantum Eigensolver. The parameters are:
        - *problem*: the problem to solve
        - *auto_setting*: if True, the parameters are automatically set
        - *num_runs*: the number of runs
        - *optimizer*: the optimizer
        - *ansatz*: the ansatz
        - *initial_point*: the initial point
        - *aggregation*: the aggregation function
        - *callback*: the callback function
        - *max_lambda_update*: the maximum lambda update if the constraints are not satisfied
        - *lambda_update_mechanism*: the lambda update mechanism among:
            - *sequential penalty increase*
            - *scaled sequential penalty increase*
            - *binary search penalty algorithm*

For each of them, the outcome is a Solution object.


Examples:
---------

Simulated Annealing
~~~~~~~~~~~~~~~~~~~
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

Quantum Annealing
~~~~~~~~~~~~~~~~~~~
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
    solution = solver.solve_dwave_quantum_annealer(
        token,
        problem,
        max_lambda_update=max_lambda_update,
        lambda_update_mechanism=lambda_update,
        lambda_strategy=lambda_strategy,
    )


Grover Adaptive Search
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from mqt.qao.constraints import Constraints
    from mqt.qao.variables import Variables
    from mqt.qao.objectivefunction import ObjectiveFunction
    from mqt.qao.problem import Problem
    from mqt.qao.solver import Solver

    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_grover_adaptive_search_qubo(
        problem, qubit_values=6, num_runs=10
    )



Quantum Approximate Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from mqt.qao.constraints import Constraints
    from mqt.qao.variables import Variables
    from mqt.qao.objectivefunction import ObjectiveFunction
    from mqt.qao.problem import Problem
    from mqt.qao.solver import Solver

    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_qaoa_qubo(
        problem,
        num_runs=10,
    )


Variational Quantum Eigensolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    variables = Variables()
    constraint = Constraints()
    a0 = variables.add_binary_variable("a")
    b0 = variables.add_binary_variable("b")
    c0 = variables.add_binary_variable("c")
    cost_function = cast(Expr, -a0 + 2 * b0 - 3 * c0 - 2 * a0 * c0 - 1 * b0 * c0)
    objective_function = ObjectiveFunction()
    objective_function.add_objective_function(cost_function)
    problem = Problem()
    problem.create_problem(variables, constraint, objective_function)
    solver = Solver()
    solution = solver.solve_vqe_qubo(
        problem,
        num_runs=10,
    )
