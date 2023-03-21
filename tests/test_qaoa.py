from mqt.problemsolver.partialcompiler.qaoa import QAOA
from qiskit import QuantumCircuit
from typing import List

def test_qaoa_init() -> None:
    q = QAOA(num_qubits=4, repetitions=3, sample_probability=0.5)
    assert isinstance(q.qc, QuantumCircuit)
    assert isinstance(q.qc_baseline, QuantumCircuit)
    assert isinstance(q.qc_compiled, QuantumCircuit)
    assert isinstance(q.to_be_checked_gates_indices, list)

def test_compile_qc() -> None:
    q = QAOA(num_qubits=4, repetitions=3, sample_probability=0.5)
    qc_baseline_compiled = q.compile_qc(baseline=True, opt_level=2)
    assert isinstance(qc_baseline_compiled, QuantumCircuit)
    qc_baseline_compiled = q.compile_qc(baseline=False, opt_level=2)
    assert isinstance(qc_baseline_compiled, QuantumCircuit)