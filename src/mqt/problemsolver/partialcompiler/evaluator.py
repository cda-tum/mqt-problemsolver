from time import time

from mqt.problemsolver.partialcompiler.qaoa import QAOA


def evaluate_QAOA(num_qubits: int = 4, repetitions: int = 3, sample_probability: float = 0.5):
    """
    Evaluate the performance of the different partial compilation methods.
    """
    q = QAOA(num_qubits=num_qubits, repetitions=repetitions, sample_probability=sample_probability)
    q.get_uncompiled_circuits()
    q.compile_qc()
    gates = q.get_to_be_checked_gates()
    start = time()
    q.set_to_be_checked_gates(gates)
    #q.reduce_swaps()
    time_new_scheme = time()-start

    start = time()
    q.compile_qc(baseline=True)
    time_baseline = time()-start

    time_ratio = time_new_scheme/time_baseline
    cx_count_ratio = q.qc_compiled.count_ops()['cx']/q.qc_compiled_baseline.count_ops()['cx']


    return (time_ratio, cx_count_ratio)
