from time import time

# from mqt.qcec import verify
from typing import TypedDict

from mqt.problemsolver.partialcompiler.qaoa import QAOA, check_gates


# @dataclass
class Result(TypedDict):
    time_ratio: float
    cx_count_ratio: float
    num_qubits: int
    num_repetitions: int
    sample_probability: float
    time_new_scheme: float
    time_baseline: float
    opt_level_baseline: int


def evaluate_QAOA(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    optimize_swaps: bool = False,
    opt_level_baseline: int = 2,
) -> Result:
    q = QAOA(num_qubits=num_qubits, repetitions=repetitions, sample_probability=sample_probability)

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc = check_gates(
        qc=qc_compiled_with_all_gates,
        remove_gates=q.remove_gates,
        to_be_checked_gates_indices=q.to_be_checked_gates_indices,
        optimize_swaps=optimize_swaps,
    )
    # if optimize_swaps:
    #     compiled_qc = reduce_swaps(qc=q.qc_compiled)
    time_new_scheme = time() - start
    # add a print function which prints the average processing time per element in q.remove_gates

    start = time()
    qc_baseline_compiled = q.compile_qc(baseline=True, opt_level=opt_level_baseline)
    time_baseline = time() - start

    time_ratio = time_new_scheme / time_baseline
    cx_count_ratio = compiled_qc.count_ops()["cx"] / qc_baseline_compiled.count_ops()["cx"]
    # try:
    #     print("QCEC:", verify(compiled_qc, qc_baseline_compiled))
    # except:
    #     print("QCEC: False")
    return Result(
        time_ratio=time_ratio,
        cx_count_ratio=cx_count_ratio,
        num_qubits=num_qubits,
        num_repetitions=repetitions,
        sample_probability=sample_probability,
        time_new_scheme=time_new_scheme,
        time_baseline=time_baseline,
        opt_level_baseline=opt_level_baseline,
    )
