from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pandas as pd
from pytket import Circuit, OpType
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import AutoRebase, BasePass, RebaseCustom
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passmanager import PassManager
from qsharp.interop.qiskit import estimate

if TYPE_CHECKING:
    from qiskit.transpiler import TransformationPass

SINGLE_QUBIT_AND_CX_QISKIT_STDGATES = [
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "cx",
]

SINGLE_QUBIT_AND_CX_TKET_STDGATES = {
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.X,
    OpType.Y,
    OpType.Z,
    OpType.H,
    OpType.S,
    OpType.Sdg,
    OpType.T,
    OpType.Tdg,
    OpType.SX,
    OpType.CX,
}


def estimate_resources(quantum_circuit: QuantumCircuit) -> tuple[int, int]:
    """
    Estimates the resource requirements of a quantum circuit.

    Args:
        quantum_circuit: The quantum circuit to estimate.

    Returns:
        A tuple containing the estimated number of physical qubits and the estimated runtime.
    """
    result = estimate(
        quantum_circuit, optimization_level=0, skip_transpilation=True
    )  # skip_transpilation to avoid version compatibility issues
    return result["physicalCounts"]["physicalQubits"], result["physicalCounts"]["runtime"]


def tk1_to_rzry(a: float, b: float, c: float) -> Circuit:
    """
    Converts a TK1 rotation gate into a sequence of RZ and RY rotations.

    Args:
        a: The angle for the RZ rotation.
        b: The angle for the RY rotation.
        c: The angle for the second RZ rotation.

    Returns:
        A Circuit object representing the equivalent rotation sequence.
    """
    circ = Circuit(1)
    if a == 0.5 and b == 0.5 and c == 0.5:
        circ.H(0)
    elif a == 0 and b == 1 and c == 0:
        circ.X(0)
    elif a == 0 and b == 1 and c == -1:
        circ.Y(0)
    elif a == 0 and b == 0 and c == 1:
        circ.Z(0)
    elif a == 0 and b == 0 and c == 0.5:
        circ.S(0)
    elif a == 0 and b == 0 and c == 3 / 2:
        circ.Sdg(0)
    elif a == 0 and b == 0 and c == 1 / 4:
        circ.T(0)
    elif a == 0 and b == 0 and c == -1 / 4:
        circ.Tdg(0)
    elif a == 0 and b == 1 / 2 and c == 0:
        circ.V(0)
    elif a == 0 and b == 3 / 2 and c == 0:
        circ.Vdg(0)
    elif a == 0 and c == 0:
        circ.Rx(b, 0)
    elif a == 1 / 2 and c == -1 / 2:
        circ.Ry(b, 0)
    elif a == 0 and b == 0:
        circ.Rz(c, 0)
    else:
        circ.Rz(c + 0.5, 0).Ry(b, 0).Rz(a - 0.5, 0)
    return circ


def cx_to_cx() -> Circuit:
    circ = Circuit(2)
    circ.CX(1, 0)
    return circ


def generate_data(
    csv_filename: pathlib.Path,
    benchmarks: list[str],
    transpiler_passes: list[BasePass | TransformationPass],
    transpiler_passes_names: list[str],
    sdk_name: str,
) -> None:
    """
    Generates and stores resource estimation data for quantum circuits after applying transpiler passes.

    This function reads quantum circuit benchmarks, applies specified transpiler passes using either Qiskit or TKET SDK,
    estimates resources before and after optimization, and saves the results to an Excel file. Only cases where the
    optimization changes the gate count, number of qubits, or runtime are recorded.

    Args:
        csv_filename: Path to the Excel file where results will be stored.
        benchmarks: List of benchmark circuit names to process.
        transpiler_passes: List of transpiler passes to apply for optimization.
        transpiler_passes_names: List of names corresponding to each transpiler pass.
        sdk_name: Name of the SDK to use ("qiskit" or "tket").

    Returns:
        None
    """
    basis_gates = SINGLE_QUBIT_AND_CX_QISKIT_STDGATES
    circuit_folder = "mqtbench"

    column_order = [
        "Benchmark",
        "Number of Qubits",
        "Transpiler Pass",
        "Original Ops",
        "Optimized Ops",
        "Gate Count (Original)",
        "Gate Count (Optimized)",
        "Physical Qubits",
        "Optimized Physical Qubits",
        "Runtime",
        "Optimized Runtime",
    ]

    df_existing = (
        pd.read_excel(csv_filename) if pathlib.Path(csv_filename).exists() else pd.DataFrame(columns=column_order)
    )

    if sdk_name == "qiskit":
        for benchmark in benchmarks:
            file_path = pathlib.Path(circuit_folder) / f"{benchmark}.qasm"

            qc = QuantumCircuit.from_qasm_file(file_path)
            transpiled_circuit = transpile(qc, basis_gates=basis_gates, optimization_level=0, seed_transpiler=0)
            num_qubits = transpiled_circuit.num_qubits
            original_ops = transpiled_circuit.count_ops()
            gate_count_original = sum(original_ops.values())

            qubits, runtime = estimate_resources(transpiled_circuit)

            for transpiler_pass in transpiler_passes:
                pass_manager = PassManager(transpiler_pass)
                optimized_circuit = pass_manager.run(transpiled_circuit)

                if transpiled_circuit != optimized_circuit:
                    optimized_ops = optimized_circuit.count_ops()

                    gate_count_optimized = sum(optimized_ops.values())
                    gate_count_diff = (gate_count_optimized - gate_count_original) / gate_count_original

                    optimized_qubits, optimized_runtime = estimate_resources(optimized_circuit)

                    relative_qubits_delta = (optimized_qubits - qubits) / qubits
                    relative_runtime_delta = (optimized_runtime - runtime) / runtime

                    if gate_count_diff != 0 or relative_qubits_delta != 0 or relative_runtime_delta != 0:
                        transpiler_pass_str = str([pass_.name() for pass_ in transpiler_pass])

                        new_data = pd.DataFrame(
                            [
                                [
                                    benchmark,
                                    num_qubits,
                                    transpiler_pass_str,
                                    original_ops,
                                    optimized_ops,
                                    gate_count_original,
                                    gate_count_optimized,
                                    qubits,
                                    optimized_qubits,
                                    runtime,
                                    optimized_runtime,
                                ]
                            ],
                            columns=column_order,
                        )

                        df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                        df_existing.to_excel(csv_filename, index=False)

    elif sdk_name == "tket":
        for benchmark in benchmarks:
            file_path = pathlib.Path(circuit_folder) / f"{benchmark}.qasm"
            read_qiskit_qc = QuantumCircuit.from_qasm_file(file_path)
            qc = qiskit_to_tk(read_qiskit_qc)

            auto_rebase_pass = AutoRebase(SINGLE_QUBIT_AND_CX_TKET_STDGATES)
            custom_rebase_pass = RebaseCustom(SINGLE_QUBIT_AND_CX_TKET_STDGATES, cx_to_cx, tk1_to_rzry)
            auto_rebase_pass.apply(qc)

            num_qubits = qc.n_qubits

            qiskit_circuit = tk_to_qiskit(qc)

            qiskit_circuit = transpile(
                qiskit_circuit, basis_gates=SINGLE_QUBIT_AND_CX_QISKIT_STDGATES, optimization_level=0, seed_transpiler=0
            )
            original_ops = qiskit_circuit.count_ops()

            gate_count_original = sum(original_ops.values())

            qubits, runtime = estimate_resources(qiskit_circuit)

            for i, transpiler_pass in enumerate(transpiler_passes):
                optimized_circuit = Circuit.from_dict(qc.to_dict())
                transpiler_pass.apply(optimized_circuit)
                custom_rebase_pass.apply(optimized_circuit)
                auto_rebase_pass.apply(optimized_circuit)

                if qc != optimized_circuit:
                    optimized_qiskit_circuit = tk_to_qiskit(optimized_circuit)
                    optimized_qiskit_circuit = transpile(
                        optimized_qiskit_circuit, basis_gates=SINGLE_QUBIT_AND_CX_QISKIT_STDGATES, optimization_level=0
                    )

                    optimized_ops = optimized_qiskit_circuit.count_ops()

                    gate_count_optimized = sum(optimized_ops.values())
                    gate_count_diff = (gate_count_optimized - gate_count_original) / gate_count_original

                    # try:
                    optimized_qubits, optimized_runtime = estimate_resources(optimized_qiskit_circuit)
                    # except TranspilerError as e:
                    #     try:
                    #         print("Removing measurements to avoid estimation errors:", e)
                    #         optimized_qiskit_circuit.remove_final_measurements()
                    #         optimized_qubits, optimized_runtime = estimate_resources(optimized_qiskit_circuit)
                    #     except BaseException as e:
                    #         print(
                    #             "Skipping circuit due to not having any magic states or measurement required for estimation:",
                    #             e,
                    #         )
                    #         continue

                    relative_qubits_delta = (optimized_qubits - qubits) / qubits
                    relative_runtime_delta = (optimized_runtime - runtime) / runtime

                    if gate_count_diff != 0 or relative_qubits_delta != 0 or relative_runtime_delta != 0:
                        transpiler_pass_str = transpiler_passes_names[i]

                        new_data = pd.DataFrame(
                            [
                                [
                                    benchmark,
                                    num_qubits,
                                    transpiler_pass_str,
                                    original_ops,
                                    optimized_ops,
                                    gate_count_original,
                                    gate_count_optimized,
                                    qubits,
                                    optimized_qubits,
                                    runtime,
                                    optimized_runtime,
                                ]
                            ],
                            columns=column_order,
                        )

                        df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                        df_existing.to_excel(csv_filename, index=False)
