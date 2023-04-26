from __future__ import annotations

from time import time
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed
from mqt.problemsolver.satellitesolver import utils
from mqt.problemsolver.satellitesolver.algorithms import solve_using_qaoa, solve_using_vqe, solve_using_w_qaoa
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer


class SatelliteResult(TypedDict):
    num_qubits: int
    calculation_time_qaoa: float
    calculation_time_wqaoa: float
    calculation_time_vqe: float
    success_rate_qaoa: float
    success_rate_wqaoa: float
    success_rate_vqe: float


def evaluate_Satellite_Solver_Noisy(num_locations: int = 5) -> SatelliteResult:
    ac_reqs = utils.init_random_acquisition_requests(num_locations)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)

    exact_mes = NumPyMinimumEigensolver()
    exact_result = MinimumEigenOptimizer(exact_mes).solve(qubo).fval

    start_time = time()
    res_qaoa = solve_using_qaoa(qubo, noisy_flag=True)
    assert res_qaoa.status.value == 0
    success_qaoa = res_qaoa.fval / exact_result
    res_qaoa_time = time() - start_time

    start_time = time()
    res_vqe = solve_using_vqe(qubo, noisy_flag=True)
    assert res_vqe.status.value == 0
    success_vqe = res_vqe.fval / exact_result
    res_vqe_time = time() - start_time

    start_time = time()
    res_w_qaoa = solve_using_w_qaoa(qubo, noisy_flag=True)
    assert res_w_qaoa.status.value == 0
    success_w_qaoa = res_w_qaoa.fval / exact_result
    res_w_qaoa_time = time() - start_time

    res = SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=res_qaoa_time,
        calculation_time_wqaoa=res_w_qaoa_time,
        calculation_time_vqe=res_vqe_time,
        success_rate_qaoa=success_qaoa,
        success_rate_vqe=success_vqe,
        success_rate_wqaoa=success_w_qaoa,
    )
    print(res)
    return res


def evaluate_Satellite_Solver(num_locations: int = 5, num_runs: int = 1) -> SatelliteResult:
    ac_reqs = utils.init_random_acquisition_requests(num_locations)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)

    exact_mes = NumPyMinimumEigensolver()
    exact_result = MinimumEigenOptimizer(exact_mes).solve(qubo).fval

    res_qaoa_times = []
    successes_qaoa = []
    for _ in range(num_runs):
        start_time = time()
        res_qaoa = solve_using_qaoa(qubo, noisy_flag=False)
        assert res_qaoa.status.value == 0
        successes_qaoa.append(res_qaoa.fval / exact_result)
        res_qaoa_times.append(time() - start_time)

    res_w_qaoa_times = []
    successes_w_qaoa = []
    for _ in range(num_runs):
        start_time = time()
        res_w_qaoa = solve_using_w_qaoa(qubo, noisy_flag=False)
        assert res_w_qaoa.status.value == 0
        successes_w_qaoa.append(res_w_qaoa.fval / exact_result)
        res_w_qaoa_times.append(time() - start_time)

    res_vqe_times = []
    successes_vqe = []
    for _ in range(num_runs):
        start_time = time()
        res_vqe = solve_using_vqe(qubo, noisy_flag=False)
        assert res_vqe.status.value == 0
        successes_vqe.append(res_vqe.fval / exact_result)
        res_vqe_times.append(time() - start_time)

    res = SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=sum(res_qaoa_times) / num_runs,
        calculation_time_wqaoa=sum(res_w_qaoa_times) / num_runs,
        calculation_time_vqe=sum(res_vqe_times) / num_runs,
        success_rate_qaoa=sum(successes_qaoa) / num_runs,
        success_rate_wqaoa=sum(successes_w_qaoa) / num_runs,
        success_rate_vqe=sum(successes_vqe) / num_runs,
    )
    print(res)

    return res


def eval_all_instances_Satellite_Solver(
    min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10, num_runs: int = 3
) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_Satellite_Solver)(i, num_runs) for i in range(min_qubits, max_qubits, stepsize)
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        "res_satellite_solver.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def eval_all_instances_Satellite_Solver_Noisy(min_qubits: int = 3, max_qubits: int = 8) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_Satellite_Solver_Noisy)(i) for i in range(min_qubits, max_qubits)
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        "res_satellite_solver_noisy.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )
