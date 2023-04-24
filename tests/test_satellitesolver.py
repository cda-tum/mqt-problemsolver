import numpy as np
import pytest
from mqt.problemsolver.satellitesolver import algorithms, utils
from mqt.problemsolver.satellitesolver.evaluator import eval_all_instances_Satellite_Solver
from mqt.problemsolver.satellitesolver.ImagingLocation import LocationRequest
from qiskit_optimization import QuadraticProgram


@pytest.fixture()
def qubo() -> QuadraticProgram:
    ac_reqs = utils.init_random_acquisition_requests(3)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)
    return qubo


def test_solve_using_qaoa(qubo: QuadraticProgram) -> None:
    res_qaoa = algorithms.solve_using_qaoa(qubo)
    assert res_qaoa is not None


def test_solve_using_wqaoa(qubo: QuadraticProgram) -> None:
    res_qaoa = algorithms.solve_using_w_qaoa(qubo)
    assert res_qaoa is not None


def test_solve_using_vqe(qubo: QuadraticProgram) -> None:
    res_qaoa = algorithms.solve_using_vqe(qubo)
    assert res_qaoa is not None


def test_eval_all_instances_Satellite_Solver() -> None:
    eval_all_instances_Satellite_Solver(min_qubits=3, max_qubits=4, stepsize=1, num_runs=1)


def test_init_random_acquisition_requests() -> None:
    req = utils.init_random_acquisition_requests(5)
    assert len(req) == 5
    assert isinstance(req[0], LocationRequest)


def test_LocationRequest() -> None:
    req = LocationRequest(np.array([1.5, 1.5, 1.5]), 5)
    assert req.imaging_attempt_score == 5
    assert req.get_longitude_angle()
    assert req.get_latitude_angle()
    assert req.get_coordinates()
