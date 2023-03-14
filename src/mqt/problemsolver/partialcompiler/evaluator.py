from time import time

from mqt.problemsolver.partialcompiler.Partial_QAOA import Partial_QAOA


def evaluate_QAOA(num_qubits: int = 4, repetitions: int = 3) -> tuple[tuple[float, int, int], tuple[float, int, int]]:
    """
    Evaluate the performance of the different partial compilation methods.
    """

    QAOA_problem_instance = Partial_QAOA(num_qubits=num_qubits, repetitions=repetitions)

    """
    Partial Compilation Method 1: (faulty because mapping of offline edges most likely changes qubit layout)
    1. Get all partially compiled circuits (without online edges) using QuantumCircuit.compose()
    2. Compile online edges
    3. Compose all circuits
    """

    (
        qc_prep_nat_gates,
        qcs_problem_nat_gates,
        qcs_mixer_nat_gates,
    ) = QAOA_problem_instance.get_partially_compiled_circuit_without_online_edges()

    start = time()
    qcs_online_edges = QAOA_problem_instance.get_compiled_online_edges()
    for i in range(QAOA_problem_instance.repetitions):
        qc_prep_nat_gates.compose(qcs_problem_nat_gates[i], inplace=True)
        qc_prep_nat_gates.compose(qcs_online_edges[i], inplace=True)
        qc_prep_nat_gates.compose(qcs_mixer_nat_gates[i], inplace=True)
    duration_partial = time() - start
    res_new_approach = (duration_partial, qc_prep_nat_gates.size(), qc_prep_nat_gates.depth())

    # Comparison to mapping the fully uncompiled circuit
    start = time()
    qc = QAOA_problem_instance.compile_full_circuit()
    duration_full = time() - start
    res_old_approach = (duration_full, qc.size(), qc.depth())

    return (res_new_approach, res_old_approach)
