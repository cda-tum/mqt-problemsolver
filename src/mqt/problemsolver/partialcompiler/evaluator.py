from time import time

# from mqt.qcec import verify
from typing import TypedDict

from mqt.problemsolver.partialcompiler.qaoa import QAOA


# @dataclass
class Result(TypedDict):
    num_qubits: int
    num_repetitions: int
    sample_probability: float
    time_baseline: float
    time_new_scheme_without_swap_opt: float
    time_new_scheme_with_swap_opt: float
    time_ratio_without_swap_opt: float
    time_ratio_with_swap_opt: float
    cx_count_ratio_without_swap_opt: float
    cx_count_ratio_with_swap_opt: float
    opt_level_baseline: int


def evaluate_QAOA(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    opt_level_baseline: int = 2,
) -> Result:
    q = QAOA(num_qubits=num_qubits, repetitions=repetitions, sample_probability=sample_probability)

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_without_swap_opt = q.check_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=False,
    )
    time_new_scheme_without_swap_opt = time() - start

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_with_swap_opt = q.check_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=True,
    )
    time_new_scheme_with_swap_opt = time() - start

    start = time()
    qc_baseline_compiled = q.compile_qc(baseline=True, opt_level=opt_level_baseline)
    time_baseline = time() - start

    time_ratio_without_swap_opt = time_new_scheme_without_swap_opt / time_baseline
    cx_count_ratio_without_swap_opt = (
        compiled_qc_without_swap_opt.count_ops()["cx"] / qc_baseline_compiled.count_ops()["cx"]
    )
    time_ratio_with_swap_opt = time_new_scheme_with_swap_opt / time_baseline
    cx_count_ratio_with_swap_opt = compiled_qc_with_swap_opt.count_ops()["cx"] / qc_baseline_compiled.count_ops()["cx"]

    # try:
    #     print("QCEC:", verify(compiled_qc, qc_baseline_compiled))
    # except:
    #     print("QCEC: False")
    return Result(
        num_qubits=num_qubits,
        num_repetitions=repetitions,
        sample_probability=sample_probability,
        time_baseline=time_baseline,
        time_ratio_without_swap_opt=time_ratio_without_swap_opt,
        time_ratio_with_swap_opt=time_ratio_with_swap_opt,
        time_new_scheme_without_swap_opt=time_new_scheme_without_swap_opt,
        time_new_scheme_with_swap_opt=time_new_scheme_with_swap_opt,
        cx_count_ratio_without_swap_opt=cx_count_ratio_without_swap_opt,
        cx_count_ratio_with_swap_opt=cx_count_ratio_with_swap_opt,
        opt_level_baseline=opt_level_baseline,
    )
