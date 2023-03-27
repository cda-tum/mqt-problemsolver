from time import time

# from mqt.qcec import verify
from typing import TypedDict

from mqt.problemsolver.partialcompiler.qaoa import QAOA


# @dataclass
class Result(TypedDict):
    num_qubits: int
    sample_probability: float
    time_baseline: float
    time_new_scheme_without_opt: float
    time_new_scheme_with_opt: float
    cx_count_baseline: float
    cx_count_without_opt: float
    cx_count_with_opt: float
    # expected_fidelity_baseline: float
    # expected_fidelity_new_scheme_without_swap_opt: float
    # expected_fidelity_new_scheme_with_swap_opt: float
    considered_following_qubits: int


def evaluate_QAOA(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    considered_following_qubits: int = 1,
) -> Result:
    q = QAOA(
        num_qubits=num_qubits,
        repetitions=repetitions,
        remove_probability=sample_probability,
        considered_following_qubits=considered_following_qubits,
    )

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_without_opt = q.check_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=False,
    )
    time_new_scheme_without_opt = time() - start
    # expected_fidelity_new_scheme_without_swap_opt = expected_fidelity(
    #     compiled_qc_without_swap_opt, device=q.backend.name().replace("fake", "ibm")
    # )

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_with_opt = q.check_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=True,
    )
    time_new_scheme_with_opt = time() - start
    # expected_fidelity_new_scheme_with_swap_opt = expected_fidelity(
    #     compiled_qc_with_swap_opt, device=q.backend.name().replace("fake", "ibm")
    # )

    qc_baseline_compiled_opt2 = q.compile_qc(baseline=True, opt_level=2)
    # expected_fidelity_baseline = expected_fidelity(
    #     qc_baseline_compiled_opt2, device=q.backend.name().replace("fake", "ibm")
    # )

    cx_count_without_opt = compiled_qc_without_opt.count_ops().get("cx")
    cx_count_with_opt = compiled_qc_with_opt.count_ops().get("cx")
    cx_count_baseline = qc_baseline_compiled_opt2.count_ops().get("cx")

    start = time()
    q.compile_qc(baseline=True, opt_level=0)
    time_baseline = time() - start

    # try:
    #     print("num_qubits:", num_qubits, "repetitions:", repetitions, "sample_probability:", sample_probability)
    #     res = verify(compiled_qc_with_swap_opt, qc_baseline_compiled)
    #     print("QCEC:", res.equivalence)
    # except Exception as e:
    #     print("QCEC: Exception", e)

    return Result(
        num_qubits=num_qubits,
        sample_probability=sample_probability,
        time_baseline=time_baseline,
        time_new_scheme_without_opt=time_new_scheme_without_opt,
        time_new_scheme_with_opt=time_new_scheme_with_opt,
        cx_count_baseline=cx_count_baseline,
        cx_count_without_opt=cx_count_without_opt,
        cx_count_with_opt=cx_count_with_opt,
        # expected_fidelity_baseline=expected_fidelity_baseline,
        # expected_fidelity_new_scheme_without_swap_opt=expected_fidelity_new_scheme_without_swap_opt,
        # expected_fidelity_new_scheme_with_swap_opt=expected_fidelity_new_scheme_with_swap_opt,
        considered_following_qubits=considered_following_qubits,
    )
