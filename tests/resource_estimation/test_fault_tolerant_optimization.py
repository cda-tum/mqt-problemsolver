from __future__ import annotations

from pathlib import Path

from pytket.passes import (
    RemoveRedundancies,
)
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
)

from mqt.problemsolver.resource_estimation.fault_tolerant_optimization import (
    SINGLE_QUBIT_AND_CX_QISKIT_STDGATES,
    generate_data_qiskit,
    generate_data_tket,
)


def test_generate_data_qiskit() -> None:
    output = Path("tests_qiskit.xlsx")
    benchmarks = [Path(__file__).parent / "inputs" / "dj_indep_qiskit_10.qasm"]
    transpiler_passes = [[Optimize1qGatesDecomposition(basis=SINGLE_QUBIT_AND_CX_QISKIT_STDGATES)]]
    generate_data_qiskit(output, benchmarks, transpiler_passes)
    output.unlink(missing_ok=True)


def test_generate_data_tket() -> None:
    output = Path("tests_tket.xlsx")
    benchmarks = [Path(__file__).parent / "inputs" / "dj_indep_qiskit_10.qasm"]
    transpiler_passes = [RemoveRedundancies()]
    transpiler_pass_names = ["RemoveRedundancies"]
    generate_data_tket(output, benchmarks, transpiler_passes, transpiler_pass_names)
    output.unlink(missing_ok=True)
