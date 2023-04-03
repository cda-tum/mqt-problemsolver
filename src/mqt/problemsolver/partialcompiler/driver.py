import numpy as np
from joblib import Parallel, delayed
from mqt.problemsolver.partialcompiler.evaluator import evaluate_QAOA


def eval_all_instances(min_qubits: int = 3, max_qubits: int = 80, stepsize: int = 10) -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3)(
        delayed(evaluate_QAOA)(i, 3, j, k)
        for i in range(min_qubits, max_qubits, stepsize)
        for j in [0.3, 0.7]
        for k in [1, 1000]
    )

    res_csv.append(list(results[0].keys()))
    for res in results:
        res_csv.append(list(res.values()))
    np.savetxt(
        "res.csv",
        res_csv,
        delimiter=",",
        fmt="%s",
    )
