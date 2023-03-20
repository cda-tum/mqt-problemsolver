from time import time

import numpy as np
from mqt.problemsolver.partialcompiler.qaoa import QAOA, check_gates, reduce_swaps


def evaluate_QAOA(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    optimize_swaps: bool = False,
    opt_level_baseline: int = 2,
) -> tuple[float, float]:
    q = QAOA(num_qubits=num_qubits, repetitions=repetitions, sample_probability=sample_probability)

    start = time()
    compiled_qc = check_gates(
        qc=q.qc_compiled, remove_gates=q.remove_gates, to_be_checked_gates_indices=q.to_be_checked_gates_indices
    )
    if optimize_swaps:
        compiled_qc = reduce_swaps(qc=q.qc_compiled)
    time_new_scheme = time() - start
    # add a print function which prints the average processing time per element in q.remove_gates

    start = time()
    qc_baseline_compiled = q.compile_qc(baseline=True, opt_level=opt_level_baseline)
    time_baseline = time() - start

    time_ratio = time_new_scheme / time_baseline
    cx_count_ratio = compiled_qc.count_ops()["cx"] / qc_baseline_compiled.count_ops()["cx"]
    # print("QCEC:", verify(q.qc_baseline, qc_baseline_compiled))
    return (np.round(time_ratio, 5), np.round(cx_count_ratio, 3))
