from __future__ import annotations

from time import time
from typing import TypedDict

import numpy as np
from joblib import Parallel, delayed

from mqt.problemsolver.partialcompiler.qaoa import QAOA


class Result(TypedDict):
    num_qubits: int
    sample_probability: float
    time_baseline_o0: float
    time_baseline_o1: float
    time_baseline_o2: float
    time_baseline_o3: float
    time_proposed: float
    cx_count_baseline_o0: float
    cx_count_baseline_o1: float
    cx_count_baseline_o2: float
    cx_count_baseline_o3: float
    cx_count_proposed: float
    considered_following_qubits: int


def evaluate_qaoa(
    num_qubits: int = 4,
    repetitions: int = 3,
    sample_probability: float = 0.5,
    considered_following_qubits: int = 1,
    satellite_use_case: bool = False,
) -> Result:
    qaoa = QAOA(
        num_qubits=num_qubits,
        repetitions=repetitions,
        sample_probability=sample_probability,
        considered_following_qubits=considered_following_qubits,
        satellite_use_case=satellite_use_case,
    )

    qc_compiled_with_all_gates = qaoa.qc_compiled.copy()
    start = time()
    compiled_qc_with_opt = qaoa.remove_unnecessary_gates(
        qc=qc_compiled_with_all_gates,
        optimize_swaps=True,
    )
    time_proposed = time() - start
    cx_count_proposed = compiled_qc_with_opt.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt0 = qaoa.compile_qc(baseline=True, opt_level=0)
    time_baseline_0 = time() - start
    cx_count_baseline_o0 = qc_baseline_compiled_opt0.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt1 = qaoa.compile_qc(baseline=True, opt_level=1)
    time_baseline_1 = time() - start
    cx_count_baseline_o1 = qc_baseline_compiled_opt1.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt2 = qaoa.compile_qc(baseline=True, opt_level=2)
    time_baseline_2 = time() - start
    cx_count_baseline_o2 = qc_baseline_compiled_opt2.count_ops().get("cx")

    start = time()
    qc_baseline_compiled_opt3 = qaoa.compile_qc(baseline=True, opt_level=3)
    time_baseline_3 = time() - start
    cx_count_baseline_o3 = qc_baseline_compiled_opt3.count_ops().get("cx")

    return Result(
        num_qubits=num_qubits,
        sample_probability=sample_probability,
        time_baseline_o0=time_baseline_0,
        time_baseline_o1=time_baseline_1,
        time_baseline_o2=time_baseline_2,
        time_baseline_o3=time_baseline_3,
        time_proposed=time_proposed,
        cx_count_baseline_o0=cx_count_baseline_o0,
        cx_count_baseline_o1=cx_count_baseline_o1,
        cx_count_baseline_o2=cx_count_baseline_o2,
        cx_count_baseline_o3=cx_count_baseline_o3,
        cx_count_proposed=cx_count_proposed,
        considered_following_qubits=considered_following_qubits,
    )


def eval_all_instances_qaoa(min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_qaoa)(i, 3, j, k)
        for i in range(min_qubits, max_qubits, stepsize)
        for j in [0.3, 0.7]
        for k in [1, 1000]
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))  # noqa: PERF401
    np.savetxt(
        "res_qaoa.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )


def eval_all_instances_satellite(min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_qaoa)(i, 3, 0.4, 1, True) for i in range(min_qubits, max_qubits, stepsize)
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))  # noqa: PERF401
    np.savetxt(
        "res_satellite.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )
