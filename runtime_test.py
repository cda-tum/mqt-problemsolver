from mqt.problemsolver.satellitesolver import utils
from mqt.problemsolver.satellitesolver.evaluator import eval_vqe_using_qiskit_runtime
from qiskit import IBMQ

IBMQ.load_account()

ac_reqs = utils.init_random_acquisition_requests(5)
mdl = utils.create_satellite_doxplex(ac_reqs)
converter, qubo = utils.convert_docplex_to_qubo(mdl)

eval_vqe_using_qiskit_runtime(5, qubo)
