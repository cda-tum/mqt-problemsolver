from time import time

from mqt.problemsolver.partialcompiler.Partial_QAOA import Partial_QAOA


def evaluate_QAOA(num_qubits: int = 4, repetitions: int = 3) -> tuple[float, float, float, float]:
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
    duration_partial_1 = time() - start

    """
    Partial Compilation Method 2: (faulty because mapping of offline edges most likely changes qubit layout)
    1. Get all partially compiled circuits (without online edges) using QuantumCircuit.append()
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
        qc_prep_nat_gates.append(
            qcs_problem_nat_gates[i], range(QAOA_problem_instance.backend.configuration().n_qubits)
        )
        qc_prep_nat_gates.append(qcs_online_edges[i], range(QAOA_problem_instance.backend.configuration().n_qubits))
        qc_prep_nat_gates.append(qcs_mixer_nat_gates[i], range(QAOA_problem_instance.backend.configuration().n_qubits))
    duration_partial_2 = time() - start

    """
    Method 3:
    1. Get all partially compiled circuits (without online edges) using QuantumCircuit.append()
    2. Get uncompiled online edges
    3. Compose all circuits
    4. Compile with mapping
    """
    (
        qc_prep_nat_gates,
        qcs_problem_nat_gates,
        qcs_mixer_nat_gates,
    ) = QAOA_problem_instance.get_partially_compiled_circuit_without_online_edges(consider_mapping=False)

    start = time()
    qcs_online_edges_uncompiled = QAOA_problem_instance.get_uncompiled_online_edges()
    for i in range(QAOA_problem_instance.repetitions):
        qc_prep_nat_gates.append(qcs_problem_nat_gates[i], range(QAOA_problem_instance.num_qubits))
        qc_prep_nat_gates.append(qcs_online_edges_uncompiled[i], range(QAOA_problem_instance.num_qubits))
        qc_prep_nat_gates.append(qcs_mixer_nat_gates[i], range(QAOA_problem_instance.num_qubits))

    QAOA_problem_instance.compile_with_mapping(qc_prep_nat_gates, opt_level=1)
    duration_partial_3 = time() - start

    # Comparison to mapping the fully uncompiled circuit
    QAOA_problem_instance.get_uncompiled_fully_composed_circuit()
    start = time()
    QAOA_problem_instance.compile_full_circuit()
    duration_full = time() - start

    return (duration_partial_1, duration_partial_2, duration_partial_3, duration_full)
