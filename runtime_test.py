import numpy as np
from mqt.problemsolver.satellitesolver import utils
from qiskit import IBMQ
from qiskit.algorithms.optimizers import SPSA
from qiskit.tools import job_monitor

IBMQ.load_account()

provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
program_id = "qaoa"
qaoa_program = provider.runtime.program(program_id)
print(f"Program name: {qaoa_program.name}, Program id: {qaoa_program.program_id}")
print(qaoa_program.parameters())


optimizer = SPSA(maxiter=100)
reps = 2
initial_point = np.random.random(2 * reps)
options = {"backend_name": "ibmq_qasm_simulator"}

num_acs = 5
ac_reqs = utils.init_random_acquisition_requests(num_acs)
mdl = utils.create_satellite_doxplex(ac_reqs)
converter, qubo = utils.convert_docplex_to_qubo(mdl)
print(qubo.prettyprint())
print(qubo.to_ising())
converter, qubo = utils.convert_docplex_to_qubo(mdl)
op = qubo.to_ising()[0]

runtime_inputs = {
    "operator": op,
    "reps": reps,
    "optimizer": optimizer,
    "initial_point": initial_point,
    "shots": 2**13,
    # Set to True when running on real backends to reduce circuit
    # depth by leveraging swap strategies. If False the
    # given optimization_level (default is 1) will be used.
    "use_swap_strategies": False,
    # Set to True when optimizing sparse problems.
    "use_initial_mapping": False,
    # Set to true when using echoed-cross-resonance hardware.
    "use_pulse_efficient": False,
}

job = provider.runtime.run(
    program_id=program_id,
    options=options,
    inputs=runtime_inputs,
)

job_monitor(job)

print(f"Job id: {job.job_id()}")
print(f"Job status: {job.status()}")
