from mqt.problemsolver.satellitesolver import utils, evaluator
from mqt.problemsolver.satellitesolver.evaluator import eval_all_instances_Satellite_Solver
from mqt.problemsolver.satellitesolver.ImagingLocation import LocationRequest
from pytest import fixture

@fixture
def qubo():
    ac_reqs = utils.init_random_acquisition_requests(3)
    mdl = utils.create_satellite_doxplex(ac_reqs)
    converter, qubo = utils.convert_docplex_to_qubo(mdl)
    return qubo

def test_solve_using_qaoa(qubo):
    res_qaoa = evaluator.solve_using_qaoa(qubo)
    assert res_qaoa is not None

def test_solve_using_wqaoa(qubo):
    res_qaoa = evaluator.solve_using_w_qaoa(qubo)
    assert res_qaoa is not None

def test_solve_using_vqe(qubo):
    res_qaoa = evaluator.solve_using_vqe(qubo)
    assert res_qaoa is not None

def test_eval_all_instances_Satellite_Solver():
    eval_all_instances_Satellite_Solver(min_qubits=3, max_qubits=4, stepsize=1, num_runs=1, noisy_flag=False)


def test_init_random_acquisition_requests():
    req = utils.init_random_acquisition_requests(5)
    assert len(req) == 5
    assert isinstance(req[0], LocationRequest)

def test_LocationRequest():
    req = LocationRequest((1.5, 1.5, 1.5), 5)
    assert req.imaging_attempt_score == 5
    assert req.get_longitude_angle()
    assert req.get_latitude_angle()
    assert req.get_coordinates()