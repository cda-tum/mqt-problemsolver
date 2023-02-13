from time import time

from mqt.problemsolver.partialcompiler.Partial_QAOA import Partial_QAOA_Instance


def evaluate_QAOA_instance(num_qubits: int = 4, repetitions: int = 3) -> tuple[float, float]:
    QAOA_problem_instance = Partial_QAOA_Instance(num_qubits=num_qubits, repetitions=repetitions)
    # Partial Evaluation
    (
        qc_prep_compiled,
        qcs_problem_compiled,
        qcs_mixer_compiled,
    ) = QAOA_problem_instance.get_partially_compiled_circuit_without_online_vertices()
    start = time()
    qc_online_edges = QAOA_problem_instance.get_compiled_online_vertices()
    for i in range(QAOA_problem_instance.repetitions):
        qc_prep_compiled.compose(qcs_problem_compiled[i], inplace=True)
        qc_prep_compiled.compose(qc_online_edges[i], inplace=True)
        qc_prep_compiled.compose(qcs_mixer_compiled[i], inplace=True)
    duration_partial = time() - start

    # Comparison
    QAOA_problem_instance.get_uncompiled_fully_composed_circuit()
    start = time()
    QAOA_problem_instance.compile_full_circuit()
    duration_full = time() - start

    return (duration_partial, duration_full)
