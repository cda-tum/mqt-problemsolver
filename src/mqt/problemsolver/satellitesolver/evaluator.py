from time import time
from typing import TypedDict

from mqt import ddsim

from mqt.problemsolver.satellitesolver import utils
from mqt.problemsolver.satellitesolver.ImagingLocation import LocationRequest

from qiskit import execute

from qiskit.algorithms.optimizers import SPSA

import time
from qiskit_optimization.problems import QuadraticProgram

import numpy as np
from joblib import Parallel, delayed

class SatelliteResult(TypedDict):
    num_qubits: int
    calculation_time_qaoa: float
    calculation_time_wqaoa: float
    success_rate_qaoa: float
    success_rate_wqaoa: float

def solve_using_w_qaoa(qubo:QuadraticProgram) -> bool:
    wqaoa = utils.W_QAOA()
    qc_wqaoa, res_wqaoa = wqaoa.get_solution(qubo)
    return res_wqaoa


def solve_using_qaoa(qubo:QuadraticProgram) -> bool:
    qaoa = utils.QAOA(QAOA_params={"reps": 3, "optimizer": SPSA(maxiter=100)})
    qc_qaoa, res_qaoa = qaoa.get_solution(qubo)

    num_shots = 10000
    backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
    job = execute(qc_qaoa, backend, shots=num_shots)
    return job.result().get_counts(qc_qaoa)


def post_process_qaoa_results(qaoa_counts, ac_reqs:list[LocationRequest], qubo:QuadraticProgram) -> tuple[bool, float]|bool :
    probs = {}
    for key in qaoa_counts:
        bin_val = bin(int(key, 2))[2:].zfill(len(qubo.variables))
        probs[bin_val] = qaoa_counts[key]

    most_likely_eigenstate_qaoa = utils.sample_most_likely(probs)
    most_likely_eigenstate_qaoa = np.array([int(s) for s in most_likely_eigenstate_qaoa], dtype=float)

    if utils.check_solution(ac_reqs, most_likely_eigenstate_qaoa):
        solution = most_likely_eigenstate_qaoa
        value = utils.calc_sol_value(ac_reqs, most_likely_eigenstate_qaoa)
        return solution, value
    else:
        return False


def evaluate_Satellite_Solver(
        num_locations: int = 5, num_runs:int=10
) -> SatelliteResult:
    ac_reqs = utils.init_random_acquisition_requests(num_locations)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)

    res_qaoa_times = []
    successes_qaoa = 0
    for j in range(num_runs):
        start_time = time.time()
        qaoa_res_raw = solve_using_qaoa(qubo)
        qaoa_res_postprocessed=post_process_qaoa_results(qaoa_res_raw, ac_reqs, qubo)
        if qaoa_res_postprocessed:
            successes_qaoa += 1
        res_qaoa_times.append(time.time() - start_time)

    res_wqaoa_times = []
    successes_wqaoa = 0
    for j in range(num_runs):
        start_time = time.time()
        res_w_qaoa = solve_using_w_qaoa(qubo)
        if res_w_qaoa.status.value == 0:
            successes_wqaoa += 1
        res_wqaoa_times.append(time.time() - start_time)

    return SatelliteResult(
        num_qubits=num_locations,
        calculation_time_qaoa=sum(res_qaoa_times) / num_runs,
        calculation_time_wqaoa=sum(res_wqaoa_times) / num_runs,
        success_rate_qaoa=successes_qaoa / num_runs,
        success_rate_wqaoa=successes_wqaoa / num_runs,
    )

def eval_all_instances_Satellite_Solver(min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_Satellite_Solver)(i) for i in range(min_qubits, max_qubits, stepsize)
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
