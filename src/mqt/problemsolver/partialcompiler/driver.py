import numpy as np
from joblib import Parallel, delayed
from mqt.problemsolver.partialcompiler.evaluator import Result, evaluate_QAOA


def eval_all_instances() -> None:
    res_csv = []
    results = Parallel(n_jobs=-1, verbose=3, backend="threading")(
        delayed(eval_single_instance)(i, 3, j, 2) for i in range(3, 80, 10) for j in [0.3, 0.5, 0.7]
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


def eval_single_instance(num_qubits: int, num_reps: int, sample_probability: float, opt_level_baseline: int) -> Result:
    return evaluate_QAOA(
        num_qubits,
        num_reps,
        sample_probability=sample_probability,
        opt_level_baseline=opt_level_baseline,
    )


eval_all_instances()
