from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Any, TypedDict

from qiskit_optimization.algorithms import MinimumEigenOptimizer

if TYPE_CHECKING:
    from qiskit.algorithms import MinimumEigensolverResult
    from qiskit_optimization.problems import QuadraticProgram

import numpy as np
from joblib import Parallel, delayed
from mqt import ddsim
from mqt.problemsolver.satellitesolver import utils
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import BackendSampler
from qiskit.providers.fake_provider import FakeMumbai


class SatelliteResult(TypedDict):
    num_qubits: int
    calculation_time_qaoa: float
    calculation_time_wqaoa: float
    calculation_time_vqe: float
    success_rate_qaoa: float
    success_rate_wqaoa: float
    success_rate_vqe: float


def solve_using_w_qaoa(qubo: QuadraticProgram, noisy_flag: bool = False) -> MinimumEigensolverResult:
    if noisy_flag:
        wqaoa = utils.W_QAOA(
            QAOA_params={"reps": 3, "optimizer": SPSA(maxiter=100), "sampler": BackendSampler(FakeMumbai())}
        )
    else:
        wqaoa = utils.W_QAOA(
            QAOA_params={
                "reps": 3,
                "optimizer": COBYLA(maxiter=100),
                "sampler": BackendSampler(ddsim.DDSIMProvider().get_backend("qasm_simulator")),
            }
        )
    qc_wqaoa, res_wqaoa = wqaoa.get_solution(qubo)
    return res_wqaoa


def solve_using_qaoa(qubo: QuadraticProgram, noisy_flag: bool = False) -> Any:
    if noisy_flag:
        qaoa = utils.QAOA(
            QAOA_params={"reps": 3, "optimizer": SPSA(maxiter=100), "sampler": BackendSampler(FakeMumbai())}
        )
    else:
        qaoa = utils.QAOA(
            QAOA_params={
                "reps": 3,
                "optimizer": COBYLA(maxiter=100),
                "sampler": BackendSampler(ddsim.DDSIMProvider().get_backend("qasm_simulator")),
            }
        )
    qc_qaoa, res_qaoa = qaoa.get_solution(qubo)
    return res_qaoa


def solve_using_vqe(qubo: QuadraticProgram, noisy_flag: bool = False) -> Any:
    if noisy_flag:
        vqe = utils.VQE(VQE_params={"optimizer": SPSA(maxiter=100), "sampler": BackendSampler(FakeMumbai())})
    else:
        vqe = utils.VQE(
            VQE_params={
                "optimizer": COBYLA(maxiter=100),
                "sampler": BackendSampler(ddsim.DDSIMProvider().get_backend("qasm_simulator")),
            }
        )
    qc_vqe, res_vqe = vqe.get_solution(qubo)
    return res_vqe


def evaluate_Satellite_Solver(num_locations: int = 5, num_runs: int = 3, noisy_flag: bool = False) -> SatelliteResult:
    ac_reqs = utils.init_random_acquisition_requests(num_locations)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)

    exact_mes = NumPyMinimumEigensolver()
    exact_result = MinimumEigenOptimizer(exact_mes).solve(qubo).fval

    res_qaoa_times = []
    successes_qaoa = 0
    for _ in range(num_runs):
        start_time = time()
        qaoa_res = solve_using_qaoa(qubo, noisy_flag)
        assert qaoa_res.status.value == 0
        successes_qaoa += qaoa_res.fval / exact_result
        res_qaoa_times.append(time() - start_time)

    res_wqaoa_times = []
    successes_wqaoa = 0
    for _ in range(num_runs):
        start_time = time()
        res_w_qaoa = solve_using_w_qaoa(qubo, noisy_flag)
        assert res_w_qaoa.status.value == 0
        successes_wqaoa += res_w_qaoa.fval / exact_result
        res_wqaoa_times.append(time() - start_time)

    res_vqe_times = []
    successes_vqe = 0
    for _ in range(num_runs):
        start_time = time()
        res_vqe = solve_using_vqe(qubo, noisy_flag)
        assert res_vqe.status.value == 0
        successes_vqe += res_vqe.fval / exact_result
        res_vqe_times.append(time() - start_time)

    res = SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=sum(res_qaoa_times) / num_runs,
        calculation_time_wqaoa=sum(res_wqaoa_times) / num_runs,
        calculation_time_vqe=sum(res_vqe_times) / num_runs,
        success_rate_qaoa=successes_qaoa / num_runs,
        success_rate_wqaoa=successes_wqaoa / num_runs,
        success_rate_vqe=successes_vqe / num_runs,
    )
    print(res)

    return res


def eval_all_instances_Satellite_Solver(
    min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10, num_runs: int = 3, noisy_flag: bool = False
) -> None:
    res_csv = []
    results = Parallel(n_jobs=8, verbose=3)(
        delayed(evaluate_Satellite_Solver)(i, num_runs, noisy_flag) for i in range(min_qubits, max_qubits, stepsize)
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        "res_satellite_solver_noise_" + str(noisy_flag) + ".csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )
