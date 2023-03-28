from time import time
from typing import TypedDict

from mqt.problemsolver.partialcompiler.qaoa import QAOA


# @dataclass
class Result(TypedDict):
    num_qubits: int
    sample_probability: float
    time_baseline_O0: float
    time_baseline_O3: float
    time_new_scheme_without_opt: float
    time_new_scheme_with_opt: float
    cx_count_baseline_O0: float
    cx_count_baseline_O3: float
    cx_count_without_opt: float
    cx_count_with_opt: float
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
    cx_count_without_opt = compiled_qc_without_opt.count_ops().get("cx")

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_with_opt = q.check_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=True,
    )
    time_new_scheme_with_opt = time() - start
    cx_count_with_opt = compiled_qc_with_opt.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt0 = q.compile_qc(baseline=True, opt_level=0)
    time_baseline_0 = time() - start
    cx_count_baseline_O0 = qc_baseline_compiled_opt0.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt3 = q.compile_qc(baseline=True, opt_level=3)
    time_baseline_3 = time() - start
    cx_count_baseline_O3 = qc_baseline_compiled_opt3.count_ops().get("cx")

    return Result(
        num_qubits=num_qubits,
        sample_probability=sample_probability,
        time_baseline_O0=time_baseline_0,
        time_baseline_O3=time_baseline_3,
        time_new_scheme_without_opt=time_new_scheme_without_opt,
        time_new_scheme_with_opt=time_new_scheme_with_opt,
        cx_count_baseline_O0=cx_count_baseline_O0,
        cx_count_baseline_O3=cx_count_baseline_O3,
        cx_count_without_opt=cx_count_without_opt,
        cx_count_with_opt=cx_count_with_opt,
        considered_following_qubits=considered_following_qubits,
    )
