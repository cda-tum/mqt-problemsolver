from time import time
from typing import TypedDict

from mqt.problemsolver.partialcompiler.qaoa import QAOA


class Result(TypedDict):
    num_qubits: int
    sample_probability: float
    time_baseline_O0: float
    time_baseline_O1: float
    time_baseline_O2: float
    time_baseline_O3: float
    time_proposed: float
    cx_count_baseline_O0: float
    cx_count_baseline_O1: float
    cx_count_baseline_O2: float
    cx_count_baseline_O3: float
    cx_count_proposed: float
    considered_following_qubits: int


def evaluate_QAOA(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    considered_following_qubits: int = 1,
    satellite_use_case: bool = False,
) -> Result:
    q = QAOA(
        num_qubits=num_qubits,
        repetitions=repetitions,
        sample_probability=sample_probability,
        considered_following_qubits=considered_following_qubits,
        satellite_use_case=satellite_use_case,
    )

    qc_compiled_with_all_gates = q.qc_compiled.copy()
    start = time()
    compiled_qc_with_opt = q.remove_unnecessary_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=True,
    )
    time_proposed = time() - start
    cx_count_proposed = compiled_qc_with_opt.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt0 = q.compile_qc(baseline=True, opt_level=0)
    time_baseline_0 = time() - start
    cx_count_baseline_O0 = qc_baseline_compiled_opt0.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt1 = q.compile_qc(baseline=True, opt_level=1)
    time_baseline_1 = time() - start
    cx_count_baseline_O1 = qc_baseline_compiled_opt1.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt2 = q.compile_qc(baseline=True, opt_level=2)
    time_baseline_2 = time() - start
    cx_count_baseline_O2 = qc_baseline_compiled_opt2.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt3 = q.compile_qc(baseline=True, opt_level=3)
    time_baseline_3 = time() - start
    cx_count_baseline_O3 = qc_baseline_compiled_opt3.count_ops().get("cx")

    return Result(
        num_qubits=num_qubits,
        sample_probability=sample_probability,
        time_baseline_O0=time_baseline_0,
        time_baseline_O1=time_baseline_1,
        time_baseline_O2=time_baseline_2,
        time_baseline_O3=time_baseline_3,
        time_proposed=time_proposed,
        cx_count_baseline_O0=cx_count_baseline_O0,
        cx_count_baseline_O1=cx_count_baseline_O1,
        cx_count_baseline_O2=cx_count_baseline_O2,
        cx_count_baseline_O3=cx_count_baseline_O3,
        cx_count_proposed=cx_count_proposed,
        considered_following_qubits=considered_following_qubits,
    )
