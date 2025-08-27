from __future__ import annotations

from time import time
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed

from mqt.problemsolver.satellitesolver import utils
from mqt.problemsolver.satellitesolver.algorithms import solve_using_qaoa, solve_using_vqe


class SatelliteResult(TypedDict):
    num_qubits: int
    calculation_time_qaoa: float
    calculation_time_vqe: float
    success_rate_qaoa: float
    success_rate_vqe: float


def evaluate_satellite_solver_noisy(num_locations: int = 5) -> SatelliteResult:
    ac_reqs = utils.init_random_location_requests(num_locations)
    qubo = utils.create_satellite_qubo(ac_reqs)

    start_time = time()
    res_qaoa = solve_using_qaoa(qubo, noisy_flag=True)
    res_qaoa_time = time() - start_time
    cost_op, _ = utils.cost_op_from_qubo(qubo)
    exact_result = utils.solve_classically(cost_op.to_matrix())
    success_qaoa = res_qaoa / exact_result

    start_time = time()
    res_vqe = solve_using_vqe(qubo, noisy_flag=True)
    success_vqe = res_vqe / exact_result
    res_vqe_time = time() - start_time

    return SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=res_qaoa_time,
        calculation_time_vqe=res_vqe_time,
        success_rate_qaoa=success_qaoa,
        success_rate_vqe=success_vqe,
    )


def evaluate_satellite_solver(num_locations: int = 5, num_runs: int = 5) -> SatelliteResult:
    ac_reqs = utils.init_random_location_requests(num_locations)
    qubo = utils.create_satellite_qubo(ac_reqs)

    exact_result = utils.solve_classically(qubo)

    res_qaoa_times = []
    successes_qaoa = []
    for _ in range(num_runs):
        start_time = time()
        res_qaoa = solve_using_qaoa(qubo, noisy_flag=False)
        successes_qaoa.append(res_qaoa / exact_result)
        res_qaoa_times.append(time() - start_time)

    res_vqe_times = []
    successes_vqe = []
    for _ in range(num_runs):
        start_time = time()
        res_vqe = solve_using_vqe(qubo, noisy_flag=False)
        successes_vqe.append(res_vqe / exact_result)
        res_vqe_times.append(time() - start_time)

    return SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=sum(res_qaoa_times) / num_runs,
        calculation_time_vqe=sum(res_vqe_times) / num_runs,
        success_rate_qaoa=sum(successes_qaoa) / num_runs,
        success_rate_vqe=sum(successes_vqe) / num_runs,
    )


def eval_all_instances_satellite_solver(
    min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10, num_runs: int = 5
) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_satellite_solver)(i, num_runs) for i in range(min_qubits, max_qubits, stepsize)
    )
    for res in results:
        assert res["success_rate_qaoa"] >= 0.3, f"QAOA success rate not 0.3 for {res}"
        assert res["success_rate_vqe"] >= 0.3, f"VQE success rate not 0.3 for {res}"
    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))  # noqa: PERF401
    np.savetxt(
        "res_satellite_solver.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def eval_all_instances_satellite_solver_noisy(min_qubits: int = 3, max_qubits: int = 8) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_satellite_solver_noisy)(i) for i in range(min_qubits, max_qubits)
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))  # noqa: PERF401
    np.savetxt(
        "res_satellite_solver_noisy.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )
