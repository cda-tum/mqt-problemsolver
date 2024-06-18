# Import of the needed libraries
from __future__ import annotations

import json
import logging
from math import ceil, log, log2, sqrt
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.model import Model
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from matplotlib import rc
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import (
    GroverOptimizer,
    MinimumEigenOptimizer,
    SolutionSample,
)
from qiskit_optimization.translators import from_docplex_mp
from qubovert import PUBO, QUBO

if TYPE_CHECKING:
    from collections.abc import Callable

    from dimod import SampleSet
    from qiskit_algorithms.optimizers import Optimizer
    from qiskit_optimization.problems import QuadraticProgram


from .problem import Problem


class Solver:
    """class of the solver to solve
    It contains object variables,
    an object constraints, an object
    cost function and an object solver.
    It creates the connection among the
    various methods for automatizing the procedure"""

    def __init__(self) -> None:
        """declaration of the needed object"""
        self.problem: Problem = Problem()
        self.pubo: PUBO = PUBO()
        self.qubo: QUBO = QUBO()
        self._number_of_lambda_update = 0

    def solve_simulated_annealing(
        self,
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
        save_compilation_time: bool = False,
    ) -> Solution | bool | None:
        """function for creating to define the solver characteristic

        Keyword arguments:
        problem -- which is the problem to solve
        auto_setting -- bool it is a flag for choosing the automatic setting of the solver parameters (False by defaults)
        beta_range -- int is the range of values assumed by the simulated annealing beta parameter (the temperature), if None the solver exploites its default option.
        num_reads -- int is the number of solver execution, 100 by default
        annealing_time -- int is the number of simulated annealing iterations
        num_sweeps_per_beta -- int define the number of iterations for which is value of beta is maintained
        beta_schedule_type -- str define the method for updating beta
        seed -- int permits to initialize the random number generator
        initial_states -- Sampleset permits to define the initial state for simulated annealing exploration
        initial_states_generator -- str permits to define the method for choosing the initial state
        max_lambda_update -- int it is the maximum number of admitted lambda update in case of not satisfied constraints
        lambda_update_mechanism -- str it is to the choosing the mechanism for updating lambda in case of not satisfied constraints
        lambda_strategy -- str it is the mechanism for selecting the lambda generation mechanisms
        save_time -- bool it is a flag for deciding if save the time required for solver execution

        Return values:
        solution -- object containing all the information about the obtained solution

        """
        if save_compilation_time:
            start = time_ns()
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            stop = time_ns()
            compilation_time = float(stop - start)
        else:
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            compilation_time = -1
        if auto_setting:  # To change with the experience
            beta_range = None
            # num_reads = 100
            annealing_time = int(10 ** (0.5 * sqrt(len(self.qubo.variables))))
            num_sweeps_per_beta = 1
            beta_schedule_type = "geometric"
            seed = None
            initial_states = None
            initial_states_generator = "random"
        if save_time:
            start = time_ns()
            samples = SimulatedAnnealingSampler().sample_qubo(
                self.qubo.Q,
                beta_range=beta_range,
                num_reads=num_reads,
                num_sweeps=annealing_time,
                num_sweeps_per_beta=num_sweeps_per_beta,
                beta_schedule_type=beta_schedule_type,
                seed=seed,
                initial_states=initial_states,
                initial_states_generator=initial_states_generator,
            )
            stop = time_ns()
            time = float(stop - start) / num_reads
        else:
            samples = SimulatedAnnealingSampler().sample_qubo(
                self.qubo.Q,
                beta_range=beta_range,
                num_reads=num_reads,
                num_sweeps=annealing_time,
                num_sweeps_per_beta=num_sweeps_per_beta,
                beta_schedule_type=beta_schedule_type,
                seed=seed,
                initial_states=initial_states,
                initial_states_generator=initial_states_generator,
            )
            time = -1.0
        sol = Solution()
        sol.create_problem(self.problem)
        solver_info: dict[str, Any] = {}
        solver_info["solver name"] = "Simulated annealer"
        solver_info["num reads"] = num_reads
        solver_info["annealing time"] = annealing_time
        if save_compilation_time:
            solver_info["compilation time"] = compilation_time
        sol.create_dwave_annealing_solution(samples, self.pubo, self.qubo, self.qubo.offset, time, solver_info)
        all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()
        print(all_satisfied)
        if not all_satisfied and self._number_of_lambda_update < max_lambda_update:
            while self._number_of_lambda_update != max_lambda_update and not all_satisfied:
                self.pubo = self.problem.update_lambda_cost_function(
                    single_satisfied, max_lambda_update, lambda_update_mechanism
                )
                self.qubo = self.pubo.to_qubo()
                if save_time:
                    start = time_ns()
                    samples = SimulatedAnnealingSampler().sample_qubo(
                        self.qubo.Q,
                        beta_range=beta_range,
                        num_reads=num_reads,
                        num_sweeps=annealing_time,
                        num_sweeps_per_beta=num_sweeps_per_beta,
                        beta_schedule_type=beta_schedule_type,
                        seed=seed,
                        initial_states=initial_states,
                        initial_states_generator=initial_states_generator,
                    )
                    stop = time_ns()
                    time = float(stop - start) / num_reads
                else:
                    samples = SimulatedAnnealingSampler().sample_qubo(
                        self.qubo.Q,
                        beta_range=beta_range,
                        num_reads=num_reads,
                        num_sweeps=annealing_time,
                        num_sweeps_per_beta=num_sweeps_per_beta,
                        beta_schedule_type=beta_schedule_type,
                        seed=seed,
                        initial_states=initial_states,
                        initial_states_generator=initial_states_generator,
                    )
                    time = -1.0
                sol = Solution()
                sol.create_problem(self.problem)
                self._number_of_lambda_update += 1
                solver_info = {}
                solver_info["solver name"] = "Simulated annealer"
                solver_info["lambda update"] = self._number_of_lambda_update
                solver_info["num reads"] = num_reads
                solver_info["annealing time"] = annealing_time
                if save_compilation_time:
                    solver_info["compilation time"] = compilation_time
                sol.create_dwave_annealing_solution(samples, self.pubo, self.qubo, self.qubo.offset, time, solver_info)
                all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()

        return sol

    def solve_dwave_quantum_annealer(
        self,
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
    ) -> Solution | bool | None:
        """function for creating to define the solver characteristic

        Keyword arguments:
        problem -- which is the problem to solve
        auto_setting -- bool it is a flag for choosing the automatic setting of the solver parameters (False by defaults)
        failover -- Signal a failover condition if a sampling error occurs
        config_file -- path of a file for configuring the solver
        endpoint: str |None = None,
        solver -- it is the name of the wanted solver
        annealing_time_scheduling -- the anneal schedule (list[list[float]]) introduce a variation  in the global anneal schedule (for reverse annealing), while the annealing time is the duration of annealing
        num_reads -- it is the number of trial
        auto_scale -- boolean for deciding if adapt the range of the problem coefficients
        flux_drift_compensation -- boolean flag indicating whether the D-Wave system compensates for flux drift.
        initial_state -- Initial state to which the system is set for reverse annealing
        programming_thermalization -- time to wait after programming the QPU for it to cool back to base temperature
        readout_thermalization -- to wait after each state is read from the QPU for it to cool back to base temperature
        reduce_intersample_correlation -- Reduces sample-to-sample correlations caused by the spin-bath polarization effect by adding a delay between reads
        max_lambda_update -- int it is the maximum number of admitted lambda update in case of not satisfied constraints
        lambda_update_mechanism -- str it is to the choosing the mechanism for updating lambda in case of not satisfied constraints
        lambda_strategy -- str it is the mechanism for selecting the lambda generation mechanisms
        save_time -- bool it is a flag for deciding if save the time required for solver execution

        Return values:
        solution -- object containing all the information about the obtained solution

        """
        if save_compilation_time:
            start = time_ns()
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            stop = time_ns()
            compilation_time = float(stop - start)
        else:
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            compilation_time = -1
        if auto_setting:  # To change with the experience
            failover = True
            config_file = None
            endpoint = None
            solver = "Advantage_system4.1"
            annealing_time_scheduling = 10 ** (0.7 * sqrt(len(self.qubo.variables)))
            if annealing_time_scheduling < 0.5:
                annealing_time_scheduling = 0.5
            elif annealing_time_scheduling > 2000:
                annealing_time_scheduling = 2000.0
            # num_reads = 100
            auto_scale = True
            flux_drift_compensation = True
            initial_state = None
            programming_thermalization = 1000.0
            readout_thermalization = 1000.0

        time = -1.0
        try:
            dwave_sampler = DWaveSampler(
                failover=failover,
                solver=solver,
                config_file=config_file,
                endpoint=endpoint,
                token=token,
            )
            dwave_sampler_embedded = EmbeddingComposite(dwave_sampler)
        except (RuntimeError, TypeError):
            print("It is not possible to exploit quantum annealer\n")
            return False
        if isinstance(annealing_time_scheduling, float):
            try:
                dwave_sampler_embedded = EmbeddingComposite(dwave_sampler)
                samples = dwave_sampler_embedded.sample_qubo(
                    self.qubo.Q,
                    num_reads=num_reads,
                    reduce_intersample_correlation=reduce_intersample_correlation,
                    annealing_time=annealing_time_scheduling,
                    auto_scale=auto_scale,
                    flux_drift_compensation=flux_drift_compensation,
                    programming_thermalization=programming_thermalization,
                    readout_thermalization=readout_thermalization,
                )
            except (RuntimeError, TypeError):
                print("It is not possible to exploit quantum annealer\n")
                return False
        else:
            try:
                samples = dwave_sampler_embedded.sample_qubo(
                    self.qubo.Q,
                    num_reads=num_reads,
                    reduce_intersample_correlation=reduce_intersample_correlation,
                    annealing_scheduling=annealing_time_scheduling,
                    auto_scale=auto_scale,
                    flux_drift_compensation=flux_drift_compensation,
                    initial_state=initial_state,
                    programming_thermalization=programming_thermalization,
                    readout_thermalization=readout_thermalization,
                )
            except (RuntimeError, TypeError):
                print("It is not possible to exploit quantum annealer\n")
                return False

        if save_time:
            time = samples.info["timing"]["qpu_anneal_time_per_sample"] * 1000  # for moving to ns

        sol = Solution()
        sol.create_problem(self.problem)
        solver_info: dict[str, Any] = {}
        solver_info["solver name"] = "Dwave annealer"
        solver_info["num reads"] = num_reads
        solver_info["annealing time"] = annealing_time_scheduling
        solver_info["qpu annealing time"] = time
        if save_compilation_time:
            solver_info["compilation time"] = compilation_time
        sol.create_dwave_annealing_solution(samples, self.pubo, self.qubo, self.qubo.offset, time, solver_info)
        all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()

        if not all_satisfied and self._number_of_lambda_update < max_lambda_update:
            while self._number_of_lambda_update != max_lambda_update and not all_satisfied:
                self.pubo = self.problem.update_lambda_cost_function(
                    single_satisfied, max_lambda_update, lambda_update_mechanism
                )
                self.qubo = self.pubo.to_qubo()
                try:
                    dwave_sampler_embedded = EmbeddingComposite(dwave_sampler)
                    samples = dwave_sampler_embedded.sample_qubo(
                        self.qubo.Q, num_reads=num_reads, reduce_intersample_correlation=reduce_intersample_correlation
                    )
                except (RuntimeError, TypeError):
                    print("It is not possible to exploit quantum annealer\n")
                    return False
                if save_time:
                    time = samples.info["qpu_anneal_time_per_sample"] * 1000  # for moving to ns
                sol = Solution()
                sol.create_problem(self.problem)
                self._number_of_lambda_update += 1
                solver_info = {}
                solver_info["solver name"] = "Dwave annealer"
                solver_info["lambda update"] = self._number_of_lambda_update
                solver_info["num reads"] = num_reads
                solver_info["annealing time"] = annealing_time_scheduling
                solver_info["qpu annealing time"] = time
                if save_compilation_time:
                    solver_info["compilation time"] = compilation_time
                sol.create_dwave_annealing_solution(samples, self.pubo, self.qubo, self.qubo.offset, time, solver_info)
                all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()
        return sol

    def solve_grover_adaptive_search_qubo(
        self,
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
    ) -> Solution | bool | None:
        """function for creating to define the solver characteristic

        Keyword arguments:
        problem -- which is the problem to solve
        auto_setting -- bool it is a flag for choosing the automatic setting of the solver parameters (False by defaults)
        simulator -- bool it is a flag for choosing the execution on a simulator or on a real devices
        backend_name -- str specifying if the name of the real or simulation backend
        IBMaccount -- str containing the IBM quantum experience account for exploiting real devices
        qubit_values -- int containing the number of qubits necessary for representing the function values. It can be derived from the bound
        coeff_precision -- float indicated the wanted precision for coefficients
        threshold -- int it is the number of consecutive positive accepted before determining that negatives are finished
        num_runs -- int it is the number of trial
        max_lambda_update -- int it is the maximum number of admitted lambda update in case of not satisfied constraints
        lambda_update_mechanism -- str it is to the choosing the mechanism for updating lambda in case of not satisfied constraints
        lambda_strategy -- str it is the mechanism for selecting the lambda generation mechanisms
        save_time -- bool it is a flag for deciding if save the time required for solver execution

        Return values:
        solution -- object containing all the information about the obtained solution

        """
        if save_compilation_time:
            start = time_ns()
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            scaled_qubo_c = round((self.qubo.copy() - self.qubo.offset) / (coeff_precision))
            min_coeff = abs(min(scaled_qubo_c.values(), key=abs))
            scaled_qubo = round(scaled_qubo_c / min_coeff)
            qp = self._from_qubovert_to_qiskit_model(scaled_qubo)
            stop = time_ns()
            compilation_time = float(stop - start)
        else:
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            scaled_qubo_c = round((self.qubo.copy() - self.qubo.offset) / (coeff_precision))
            min_coeff = abs(min(scaled_qubo_c.values(), key=abs))
            scaled_qubo = round(scaled_qubo_c / min_coeff)
            qp = self._from_qubovert_to_qiskit_model(scaled_qubo)
            compilation_time = -1
        logging.getLogger("qiskit").setLevel(logging.ERROR)
        res = []

        if auto_setting:  # To change with the experience
            # num_runs = 100
            threshold = 2 * len(self.qubo.variables)
            qubit_values = ceil(log2(abs(self.problem.upper_lower_bound_posiform_and_negaform_method(scaled_qubo))))
        elif boundaries_estimation_method == "upper lower bound posiform and negaform method":
            qubit_values = ceil(log2(abs(self.problem.upper_lower_bound_posiform_and_negaform_method(scaled_qubo))))
        elif boundaries_estimation_method == "naive":
            qubit_values = ceil(log2(abs(self.problem.upper_lower_bound_naive_method(scaled_qubo))))

        grover_optimizer = GroverOptimizer(qubit_values, num_iterations=threshold, sampler=Sampler())

        if save_time:
            start = time_ns()
            for _run in range(num_runs):
                results = grover_optimizer.solve(qp)
                res.append(results)
            stop = time_ns()
            time = float(stop - start) / num_runs
        else:
            for _run in range(num_runs):
                results = grover_optimizer.solve(qp)
                res.append(results)
            time = -1.0

        sol = Solution()
        sol.create_problem(self.problem)
        solver_info: dict[str, Any] = {}
        solver_info["solver name"] = "GAS qubo"
        solver_info["qubit values"] = qubit_values
        solver_info["num runs"] = num_runs
        solver_info["threshold"] = threshold
        solver_info["time"] = time
        if save_compilation_time:
            solver_info["compilation time"] = compilation_time
        sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
        all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()

        if not all_satisfied and self._number_of_lambda_update < max_lambda_update:
            while self._number_of_lambda_update != max_lambda_update and not all_satisfied:
                self.pubo = self.problem.update_lambda_cost_function(
                    single_satisfied, max_lambda_update, lambda_update_mechanism
                )
                self.qubo = self.pubo.to_qubo()
                scaled_qubo_c = round((self.qubo.copy() - self.qubo.offset) / (coeff_precision))
                min_coeff = abs(min(scaled_qubo_c.values(), key=abs))
                scaled_qubo = round(scaled_qubo_c / min_coeff)
                qp = self._from_qubovert_to_qiskit_model(scaled_qubo)

                if auto_setting or boundaries_estimation_method == "upper lower bound posiform and negaform method":
                    qubit_values = ceil(
                        log2(abs(self.problem.upper_lower_bound_posiform_and_negaform_method(scaled_qubo)))
                    )
                elif boundaries_estimation_method == "naive":
                    qubit_values = ceil(log2(abs(self.problem.upper_lower_bound_naive_method(scaled_qubo))))

                grover_optimizer = GroverOptimizer(qubit_values, num_iterations=threshold, sampler=Sampler())

                if save_time:
                    start = time_ns()
                    for _run in range(num_runs):
                        results = grover_optimizer.solve(qp)
                        res.append(results)
                    stop = time_ns()
                    time = float(stop - start) / num_runs
                else:
                    for _run in range(num_runs):
                        results = grover_optimizer.solve(qp)
                        res.append(results)
                    time = -1.0
                sol = Solution()
                sol.create_problem(self.problem)
                self._number_of_lambda_update += 1
                solver_info = {}
                solver_info["solver name"] = "GAS qubo"
                solver_info["qubit values"] = qubit_values
                solver_info["lambda update"] = self._number_of_lambda_update
                solver_info["num runs"] = num_runs
                solver_info["threshold"] = threshold
                solver_info["time"] = time
                if save_compilation_time:
                    solver_info["compilation time"] = compilation_time
                sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
                all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()
        return sol

    def solve_qaoa_qubo(
        self,
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
    ) -> Solution | bool | None:
        """function for creating to define the solver characteristic

        Keyword arguments:
        problem -- which is the problem to solve
        auto_setting -- bool it is a flag for choosing the automatic setting of the solver parameters (False by defaults)
        simulator -- bool it is a flag for choosing the execution on a simulator or on a real devices
        backend_name -- str specifying if the name of the real or simulation backend
        IBMaccount -- str containing the IBM quantum experience account for exploiting real devices
        classical_optimizer -- str identifying the wanted classical optimizer
        num_runs -- int it is the number of trial
        max_lambda_update -- int it is the maximum number of admitted lambda update in case of not satisfied constraints
        lambda_update_mechanism -- str it is to the choosing the mechanism for updating lambda in case of not satisfied constraints
        lambda_strategy -- str it is the mechanism for selecting the lambda generation mechanisms
        save_time -- bool it is a flag for deciding if save the time required for solver execution

        Return values:
        solution -- object containing all the information about the obtained solution

        """
        if save_compilation_time:
            start = time_ns()
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            qp = self._from_qubovert_to_qiskit_model(self.qubo)
            stop = time_ns()
            compilation_time = float(stop - start)
        else:
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            qp = self._from_qubovert_to_qiskit_model(self.qubo)
            compilation_time = -1
        logging.getLogger("qiskit").setLevel(logging.ERROR)
        res = []
        if auto_setting:  # To change with the experience
            reps = ceil(2 * sqrt(len(self.qubo.variables)))
            if reps == 0:
                reps = 1
        if mixer is None or initial_state is None:
            beta = Parameter("β")
            var_num = len(self.qubo.variables)
            mixer = QuantumCircuit(var_num)
            for _idx in range(var_num):
                mixer.rx(2 * beta, _idx)

            initial_state = QuantumCircuit(var_num)
            for _idx in range(var_num):
                initial_state.h(_idx)

        if optimizer is None:
            optimizer = COBYLA()
        qaoa_mes = QAOA(
            sampler=Sampler(),
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state,
            mixer=mixer,
            initial_point=initial_point,
            aggregation=aggregation,
            callback=callback,
        )
        qaoa = MinimumEigenOptimizer(qaoa_mes)

        if save_time:
            start = time_ns()
            for _run in range(num_runs):
                results = qaoa.solve(qp)
                res.append(results)
            stop = time_ns()
            time = float(stop - start) / num_runs
        else:
            for _run in range(num_runs):
                results = qaoa.solve(qp)
                res.append(results)
            time = -1.0

        sol = Solution()
        sol.create_problem(self.problem)
        solver_info: dict[str, Any] = {}
        solver_info["solver name"] = "QAOA qubo"
        solver_info["num runs"] = num_runs
        solver_info["repetitions"] = reps
        solver_info["time"] = time
        if save_compilation_time:
            solver_info["compilation time"] = compilation_time
        sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
        all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()

        if not all_satisfied and self._number_of_lambda_update < max_lambda_update:
            while self._number_of_lambda_update != max_lambda_update and not all_satisfied:
                self.pubo = self.problem.update_lambda_cost_function(
                    single_satisfied, max_lambda_update, lambda_update_mechanism
                )
                self.qubo = self.pubo.to_qubo()
                qp = self._from_qubovert_to_qiskit_model(self.qubo)

                if save_time:
                    start = time_ns()
                    for _run in range(num_runs):
                        results = qaoa.solve(qp)
                        res.append(results)
                    stop = time_ns()
                    time = float(stop - start) / num_runs
                else:
                    for _run in range(num_runs):
                        results = qaoa.solve(qp)
                        res.append(results)
                    time = -1.0
                sol = Solution()
                sol.create_problem(self.problem)
                self._number_of_lambda_update += 1
                solver_info = {}
                solver_info["solver name"] = "QAOA qubo"
                solver_info["lambda update"] = self._number_of_lambda_update
                solver_info["num runs"] = num_runs
                solver_info["repetitions"] = reps
                solver_info["time"] = time
                if save_compilation_time:
                    solver_info["compilation time"] = compilation_time
                sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
                all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()
        return sol

    def solve_vqe_qubo(
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
    ) -> Solution | bool | None:
        """function for creating to define the solver characteristic

        Keyword arguments:
        problem -- which is the problem to solve
        auto_setting -- bool it is a flag for choosing the automatic setting of the solver parameters (False by defaults)
        simulator -- bool it is a flag for choosing the execution on a simulator or on a real devices
        backend_name -- str specifying if the name of the real or simulation backend
        IBMaccount -- str containing the IBM quantum experience account for exploiting real devices
        classical_optimizer -- str identifying the wanted classical optimizer
        num_runs -- int it is the number of trial
        max_lambda_update -- int it is the maximum number of admitted lambda update in case of not satisfied constraints
        lambda_update_mechanism -- str it is to the choosing the mechanism for updating lambda in case of not satisfied constraints
        lambda_strategy -- str it is the mechanism for selecting the lambda generation mechanisms
        save_time -- bool it is a flag for deciding if save the time required for solver execution

        Return values:
        solution -- object containing all the information about the obtained solution

        """
        if save_compilation_time:
            start = time_ns()
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            qp = self._from_qubovert_to_qiskit_model(self.qubo)
            stop = time_ns()
            compilation_time = float(stop - start)
        else:
            self.problem = problem
            self.pubo = self.problem.write_the_final_cost_function(lambda_strategy, lambda_value=lambda_value)
            self.qubo = self.pubo.to_qubo()
            qp = self._from_qubovert_to_qiskit_model(self.qubo)
            compilation_time = -1
        logging.getLogger("qiskit").setLevel(logging.ERROR)
        res = []
        if auto_setting or ansatz is None:
            ansatz = TwoLocal(num_qubits=len(self.qubo.variables), rotation_blocks="ry", entanglement_blocks="cz")

        if optimizer is None:
            optimizer = COBYLA()

        vqe_mes = SamplingVQE(
            sampler=Sampler(),
            optimizer=optimizer,
            ansatz=ansatz,
            initial_point=initial_point,
            aggregation=aggregation,
            callback=callback,
        )
        vqe = MinimumEigenOptimizer(vqe_mes)

        if save_time:
            start = time_ns()
            for _run in range(num_runs):
                results = vqe.solve(qp)
                res.append(results)
            stop = time_ns()
            time = float(stop - start) / num_runs
        else:
            for _run in range(num_runs):
                results = vqe.solve(qp)
                res.append(results)
            time = -1.0

        sol = Solution()
        sol.create_problem(self.problem)
        solver_info: dict[str, Any] = {}
        solver_info["solver name"] = "VQE qubo"
        solver_info["num runs"] = num_runs
        solver_info["time"] = time
        if save_compilation_time:
            solver_info["compilation time"] = compilation_time
        sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
        all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()

        if not all_satisfied or self._number_of_lambda_update < max_lambda_update:
            while self._number_of_lambda_update != max_lambda_update and not all_satisfied:
                self.pubo = self.problem.update_lambda_cost_function(
                    single_satisfied, max_lambda_update, lambda_update_mechanism
                )
                self.qubo = self.pubo.to_qubo()
                qp = self._from_qubovert_to_qiskit_model(self.qubo)

                if save_time:
                    start = time_ns()
                    for _run in range(num_runs):
                        results = vqe.solve(qp)
                        res.append(results)
                    stop = time_ns()
                    time = float(stop - start) / num_runs
                else:
                    for _run in range(num_runs):
                        results = vqe.solve(qp)
                        res.append(results)
                    time = -1.0
                sol = Solution()
                sol.create_problem(self.problem)
                self._number_of_lambda_update += 1
                solver_info = {}
                solver_info["solver name"] = "VQE qubo"
                solver_info["lambda update"] = self._number_of_lambda_update
                solver_info["num runs"] = num_runs
                solver_info["time"] = time
                if save_compilation_time:
                    solver_info["compilation time"] = compilation_time
                sol.create_qiskit_qubo_solution(res, self.pubo, self.qubo, time, solver_info)
                all_satisfied, single_satisfied = sol.check_constraint_optimal_solution()
            return sol
        return None

    def get_lambda_updates(self) -> int:
        return self._number_of_lambda_update

    @staticmethod
    def _from_qubovert_to_qiskit_model(qubo: QUBO) -> QuadraticProgram:
        """function for converting the qubovert formulation into the qiskit quadratic model one

        Keyword arguments:
        qubo -- it is the qubo formulation of the problem of interest

        Return values:
        QuadraticProgram -- qiskit compliant formulation
        """
        dpmodel = Model()
        docplex_variables = {var: dpmodel.binary_var(name=f"x{var}") for var in qubo.variables}
        objective_expr = 0
        min(qubo.values(), key=abs)
        for key in qubo:
            if len(key) == 2:
                objective_expr += qubo[key] * docplex_variables[key[0]] * docplex_variables[key[1]]
            elif len(key) == 1:
                objective_expr += qubo[key] * docplex_variables[key[0]]
        dpmodel.minimize(objective_expr)
        return from_docplex_mp(dpmodel)


class Solution:
    """class of the solution
    It contains the solutions, the energy value
    and can provide some statistics"""

    def __init__(self) -> None:
        """declaration of the needed object"""
        self.problem: Problem = Problem()
        self.energies: list[float] = []
        self.best_energy: float = 0.0
        self.cost_functions_best_energy: dict[str, float] = {}
        self.constraints_satisfy: bool = False
        self.single_constraint_satisfy: list[bool] = []
        self.best_solution: dict[str, Any] = {}
        self.best_solution_original_var: dict[str, Any] = {}
        self.solutions_original_var: list[dict[str, Any]] = []
        self.solutions: list[dict[str, Any]] = []
        self.time: float
        self.valid_solution_rate: dict[int, float] = {}
        self.solver_info: dict[str, Any] = {}
        self.solve_qubo: bool = True
        self.pubo: PUBO
        self.qubo: QUBO

    def create_problem(self, problem: Problem) -> None:
        """function for saving problem information

        Keyword arguments:
        problem -- which is the problem to solve

        Return values:
        None

        """
        self.problem = problem

    def optimal_solution_cost_functions_values(self) -> dict[str, float] | bool:
        """function for computing the value associated with each cost function with the optimal solution found

        Keyword arguments:

        Return values:
        dict[str, float] -- containing the cost function-value couples

        """
        temp = self.problem.objective_function.substitute_values(self.best_solution, self.problem.variables)
        if not isinstance(temp, bool):
            self.cost_functions_best_energy = temp
        return self.cost_functions_best_energy

    def check_constraint_optimal_solution(self, weak: bool = False) -> tuple[bool, list[bool]]:
        """function for computing the value associated with each cost function with the optimal solution found

        Keyword arguments:

        Return values:
        constraints_satisfy -- bool which is equal to True if all the constraints are satisfied
        single_constraint_satisfy -- list of bool, saying which are the satisfied constraint and which are not

        """
        self.constraints_satisfy, self.single_constraint_satisfy = self.problem.constraints.check_constraint(
            self.best_solution, self.best_solution_original_var, self.problem.variables, weak
        )
        return self.constraints_satisfy, self.single_constraint_satisfy

    def check_constraint_all_solutions(self, weak: bool = False) -> tuple[list[bool], list[list[bool]]]:
        """function for computing the value associated with each cost function with all the solution found

        Keyword arguments:

        Return values:
        all_satisfied -- list bool where each element is equal to True if all the constraints are satisfied
        single_satisfied -- list of list of bool, saying which are the satisfied constraint and which are not for each solution

        """
        all_satisfied = []
        single_satisfied = []
        for j in range(len(self.solutions)):
            all_c, single = self.problem.constraints.check_constraint(
                self.solutions[j], self.solutions_original_var[j], self.problem.variables, weak
            )
            all_satisfied.append(all_c)
            single_satisfied.append(single)
        return all_satisfied, single_satisfied

    def create_dwave_annealing_solution(
        self,
        samples: SampleSet,
        pubo: PUBO,
        qubo: QUBO,
        offset: float = 0,
        time: float = -1.0,
        solver_info: dict[str, Any] | None = None,
    ) -> None:
        """function for characterizing the solution object if the chosen solver was the dwave annealing

        Keyword arguments:
        samples -- sampleset object provided as outcome by the simulated annealing
        pubo -- PUBO, it is the solved PUBO problem
        offset -- float, it is the problem offset
        time -- float, it is the solver execution time

        Return values:
        None

        """
        if solver_info is None:
            solver_info = {}
        self.qubo = qubo
        self.pubo = pubo
        self.solver_info = solver_info
        for sample in samples.record:
            for _oc in range(sample[2]):
                self.energies.append(sample[1] + offset)
        self.best_energy = samples.first.energy + offset

        for sample in list(samples.samples()):
            converted_sol = self.problem.variables.convert_simulated_annealing_solution(pubo.convert_solution(sample))
            if isinstance(converted_sol, dict):
                self.solutions.append(converted_sol)
            sol = pubo.convert_solution(sample)
            temp = {}
            for var in sol:
                temp[var] = float(sol[var])
            self.solutions_original_var.append(temp)
        converted_sol = self.problem.variables.convert_simulated_annealing_solution(
            pubo.convert_solution(samples.first.sample)
        )
        sol = pubo.convert_solution(samples.first.sample)
        for var in sol:
            self.best_solution_original_var[var] = float(sol[var])

        if isinstance(converted_sol, dict):
            self.best_solution = converted_sol
        self.time = time

    def create_qiskit_qubo_solution(
        self,
        res: list[SolutionSample],
        pubo: PUBO,
        qubo: QUBO,
        time: float = -1.0,
        solver_info: dict[str, Any] | None = None,
    ) -> None:
        """function for characterizing the solution object if the chosen solver was the dwave annealing

        Keyword arguments:
        samples -- sampleset object provided as outcome by the simulated annealing
        pubo -- PUBO, it is the solved PUBO problem
        time -- float, it is the solver execution time

        Return values:
        None

        """
        if solver_info is None:
            solver_info = {}
        self.solver_info = solver_info
        first: bool = True
        val: float = 0.0
        self.qubo = qubo
        self.pubo = pubo
        for elem in res:
            s = {}
            if isinstance(elem.x, np.ndarray):
                for i in range(len(elem.x)):
                    s[i] = elem.x[i]
                sol = pubo.convert_solution(s)
                converted_sol = self.problem.variables.convert_simulated_annealing_solution(sol)
                self.solutions_original_var.append(sol)
                if isinstance(converted_sol, dict):
                    self.solutions.append(converted_sol)
                val = pubo.value(sol)
                self.energies.append(val)
                if first:
                    self.best_energy = val
                    self.best_solution_original_var = sol
                    if isinstance(converted_sol, dict):
                        self.best_solution = converted_sol
                    first = False
                elif val < self.best_energy:
                    self.best_energy = val
                    self.best_solution_original_var = sol
                    if isinstance(converted_sol, dict):
                        self.best_solution = converted_sol
        self.time = time

    def _create_qiskit_pubo_solution(
        self,
        res: list[tuple[dict[str, float], float]],
        pubo: PUBO,
        time: float = -1.0,
        solver_info: dict[str, Any] | None = None,
    ) -> None:
        """function for characterizing the solution object if the chosen solver was the dwave annealing

        Keyword arguments:
        samples -- list[tuple[dict[str, int], float]] object provided as outcome by the simulated annealing
        pubo -- PUBO, it is the solved PUBO problem
        time -- float, it is the solver execution time

        Return values:
        None

        """
        if solver_info is None:
            solver_info = {}
        self.solver_info = solver_info
        self.solve_qubo = False
        self.pubo = pubo
        first: bool = True
        val: float = 0.0
        for elem in res:
            sol = elem[0]
            converted_sol = self.problem.variables.convert_simulated_annealing_solution(sol)
            self.solutions_original_var.append(sol)
            if isinstance(converted_sol, dict):
                self.solutions.append(converted_sol)
            val = pubo.value(sol)
            self.energies.append(val)
            if first:
                self.best_energy = val
                self.best_solution_original_var = sol
                if isinstance(converted_sol, dict):
                    self.best_solution = converted_sol
                first = False
            elif val < self.best_energy:
                self.best_energy = val
                self.best_solution_original_var = sol
                if isinstance(converted_sol, dict):
                    self.best_solution = converted_sol
        self.time = time

    def show_cumulative(
        self, save: bool = False, show: bool = True, filename: str = "", label: str = "", latex: bool = False
    ) -> None:
        """function for showing the cumulative distribution of the results

        Keyword arguments:
        save -- bool, it is equal to true if the user wants to save it in a picture
        show -- bool, it is equal to true if the user wants simply to look the cumulative -- default option
        filename -- str, it is the name the user wants to give to the saved cumulative picture
        label -- str, it is the eventual label to the cumulative in the legend
        latex -- bool, it is equal to true if the user wants the plot labels in latex font

        Return values:
        None

        """
        if latex:
            rc("text", usetex=True)
            plt.rc("text", usetex=True)
            if label:
                plt.hist(
                    self.energies,
                    cumulative=True,
                    histtype="step",
                    linewidth=2,
                    bins=100,
                    label=r"\textit{" + label + "}",
                )
            else:
                plt.hist(self.energies, cumulative=True, histtype="step", linewidth=2, bins=100)
            plt.title(r"\textbf{Cumulative distribution}", fontsize=20)
            plt.xlabel(r"\textit{Energy}", fontsize=20)
            plt.ylabel(r"\textit{occurrence}", fontsize=20)
        else:
            if label:
                plt.hist(self.energies, cumulative=True, histtype="step", linewidth=2, bins=100, label=label)
            else:
                plt.hist(self.energies, cumulative=True, histtype="step", linewidth=2, bins=100)
            plt.title("Cumulative distribution", fontsize=20)
            plt.xlabel("Energy", fontsize=20)
            plt.ylabel("occurrence", fontsize=20)
        if label:
            leg = plt.legend(loc="lower right", frameon=True, fontsize=15)
            leg.get_frame().set_facecolor("white")
        if save:
            if not filename:
                print("The file name field is empty. Default name is given to the file\n")
                plt.savefig("Test.eps", format="eps")
                plt.savefig("Test.png", format="png")
                plt.savefig("Test.pdf", format="pdf")
                plt.close()
            else:
                plt.savefig(filename + ".eps", format="eps")
                plt.savefig(filename + ".png", format="png")
                plt.savefig(filename + ".pdf", format="pdf")
                plt.close()
        elif show:
            plt.show()

    def valid_solutions(self, weak: bool = True) -> float:
        """function for evaluating the rate of valid solution and the amount of violations

        Keyword arguments:
        weak -- bool, if it is equal to True the weak constraint satisfability is also considered

        Return values:
        float -- rate of valid solution
        """
        all_satisfied, single_satisfied = self.check_constraint_all_solutions(weak=weak)
        self.valid_solution_rate[0] = 0
        if len(all_satisfied) != 0:
            for j in range(len(all_satisfied)):
                if all_satisfied[j]:
                    if 0 in self.valid_solution_rate:
                        self.valid_solution_rate[0] += 1
                else:
                    count = 0
                    for k in range(len(single_satisfied[j])):
                        if not single_satisfied[j][k]:
                            count += 1
                    if count in self.valid_solution_rate:
                        self.valid_solution_rate[count] += 1
                    else:
                        self.valid_solution_rate[count] = 1
            for key in self.valid_solution_rate:
                self.valid_solution_rate[key] /= len(all_satisfied)
        return self.valid_solution_rate[0]

    def p_range(self, ref_value: float | None = None) -> float:
        """function for  computing the probability of obtaining a value below the reference energy. If the reference energy is not provided, the outcome become the probability of obtaining the lower energy obtained


        Keyword arguments:
        ref_value -- flot is the reference value

        Return values:
        float -- probability of obtaining a value lower than the reference
        """
        if ref_value is None:
            ret = self.energies.count(self.best_energy) / len(self.energies)
        else:
            count = 0
            for val in self.energies:
                if val <= ref_value:
                    count += 1

            ret = count / len(self.energies)
        return ret

    def tts(self, ref_value: float | None = None, target_probability: float = 0.99) -> float:
        """function for  computing the Time-To_Solution of obtaining a wanted results with a certain probability


        Keyword arguments:
        ref_value -- float is the reference value
        target_probability -- float is the desired probability of obtaining value lower than reference

        Return values:
        float -- probability of obtaining a value lower than the reference
        """
        p = self.p_range(ref_value)
        if self.time > 0:
            if p == 0:
                print(
                    "The reference value or values lower than it were not obtained. It is not possible to compute TTS\n"
                )
                t = -1.0
            elif p == 1:
                t = self.time
            else:
                try:
                    t = self.time * (log(1 - target_probability) / log(1 - p))
                except ZeroDivisionError:
                    return -1

        else:
            print("The execution time is not available. It is not possible to compute TTS\n")
            t = -1.0
        return t

    def wring_json_reports(
        self,
        filename: str = "report",
        weak: bool = False,
        ref_value: float | None = None,
        target_probability: float = 0.99,
        problem_features: bool = False,
    ) -> None:
        """function for obtaining a complete report about solution and eventually problems transformation

        Keyword arguments:
        filename -- str name for the report file
        weak -- bool for defining if consider the weak constraints in the check constrains
        ref_value -- float reference optimal value
        target_probability -- float for computing tts
        problem features -- bool for deciding if save the problem transformation info

        Return values:
        None
        """
        data: dict[str, Any] = {}
        for info in self.solver_info:
            data[info] = self.solver_info[info]
        data["energies"] = self.energies
        data["best energy"] = self.best_energy
        if self.cost_functions_best_energy == {}:
            self.optimal_solution_cost_functions_values()
        data["cost functions energy best solution"] = self.cost_functions_best_energy
        self.check_constraint_optimal_solution(weak)
        data["constraint satisfaction best solution"] = self.constraints_satisfy
        data["best solution"] = self.best_solution
        data["solutions"] = self.solutions
        data["best solution binary"] = self.best_solution_original_var
        data["solutions binary"] = self.solutions_original_var
        if self.time > 0:
            data["time"] = self.time
        if self.valid_solution_rate == {}:
            self.valid_solutions(weak)
        data["valid solution rate"] = self.valid_solution_rate
        if ref_value is None:
            ref_value = self.best_energy
        p = self.p_range(ref_value)
        if p > 0:
            data["p range"] = p
        tts_v = self.tts(ref_value, target_probability)
        if tts_v > 0:
            data["TTS"] = tts_v
        if problem_features:
            num_elm: dict[int, int]
            if self.solve_qubo:
                q: dict[str, float] = {}
                for key in list(self.qubo.keys()):
                    q[str(key)] = self.qubo[key]
                data["qubo"] = q
                data["qubo num var"] = len(self.qubo.variables)
                num_elm = {}
                elems: dict[int, list[float]] = {}
                coeff = []
                for key in self.qubo:
                    if len(key) in num_elm:
                        num_elm[len(key)] += 1
                        elems[len(key)].append(self.pubo[key])
                    else:
                        num_elm[len(key)] = 1
                        elems[len(key)] = []
                        elems[len(key)].append(self.pubo[key])
                    coeff.append(self.pubo[key])
                elems_m = {}
                elems_v = {}
                coeff_m = sum(coeff) / len(coeff)
                coeff_v = np.var(coeff)
                for key, value in elems.items():
                    elems_m[key] = sum(value) / len(value)
                    elems_v[key] = np.var(value)
                data["qubo contributions"] = num_elm
                data["qubo contributions average"] = elems_m
                data["qubo contributions variance"] = elems_v
                data["qubo coefficient average"] = coeff_m
                data["qubo coefficient variance"] = coeff_v
            num_elm = {}
            elems = {}
            coeff = []
            for key in self.pubo:
                if len(key) in num_elm:
                    num_elm[len(key)] += 1
                    elems[len(key)].append(self.pubo[key])
                else:
                    num_elm[len(key)] = 1
                    elems[len(key)] = []
                    elems[len(key)].append(self.pubo[key])
                coeff.append(self.pubo[key])
            pu: dict[str, float] = {}
            elems_m = {}
            elems_v = {}
            coeff_m = sum(coeff) / len(coeff)
            coeff_v = np.var(coeff)
            for key in list(self.pubo.keys()):
                pu[str(key)] = self.pubo[key]
            for key in elems:
                elems_m[key] = sum(elems[key]) / len(elems[key])
                elems_v[key] = np.var(elems[key])
            data["pubo"] = pu
            data["num var"] = len(self.pubo.variables)
            data["pubo contributions"] = num_elm
            data["pubo contributions average"] = elems_m
            data["pubo contributions variance"] = elems_v
            data["pubo coefficient average"] = coeff_m
            data["pubo coefficient variance"] = coeff_v
            data["lambdas"] = self.problem.lambdas

        file_path = Path(filename + ".json")
        with file_path.open("w") as outfile:
            json.dump(data, outfile)
