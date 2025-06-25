import os
import time

import pandas as pd
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes.optimization import *

from qsharp.interop.qiskit import estimate

from pytket import Circuit, OpType
from pytket.passes import AutoRebase, RebaseCustom
from pytket.qasm import circuit_from_qasm
from pytket.extensions.qiskit import tk_to_qiskit

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
    OpType.CX
}


def estimate_resources(quantum_circuit):
    result = estimate(quantum_circuit, optimization_level=0)
    return result['physicalCounts']['physicalQubits'], result['physicalCounts']['runtime']

def tk1_to_rzry(a, b, c):
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
    elif a == 0 and b == 0 and c == 3/2:
        circ.Sdg(0)
    elif a == 0 and b == 0 and c == 1/4:
        circ.T(0)
    elif a == 0 and b == 0 and c == -1/4:
        circ.Tdg(0)
    elif a == 0 and b == 1/2 and c == 0:
        circ.V(0)
    elif a == 0 and b == 3/2 and c == 0:
        circ.Vdg(0)
    elif a == 0 and c == 0:
        circ.Rx(b, 0)
    elif a == 1/2 and c == -1/2:
        circ.Ry(b, 0)
    elif a == 0 and b == 0:
        circ.Rz(c, 0)
    else:
        circ.Rz(c + 0.5, 0).Ry(b, 0).Rz(a - 0.5, 0)
    return circ

def cx_to_cx():
    circ = Circuit(2)
    circ.CX(1,0)
    return circ

def generate_data(csv_filename, benchmarks, transpiler_passes, transpiler_passes_names, sdk_name):

    basis_gates = SINGLE_QUBIT_AND_CX_QISKIT_STDGATES
    if sdk_name == "qiskit":
        circuit_folder = "MQTBenchQiskit"
    elif sdk_name == "tket":
        circuit_folder = "MQTBenchTket"

    column_order = [
        "Benchmark", "Number of Qubits", "Transpiler Pass",
        "Original Ops", "Optimized Ops",
        "Gate Count (Original)", "Gate Count (Optimized)",
        "Physical Qubits", "Optimized Physical Qubits",
        "Runtime", "Optimized Runtime",
        "Optimization Time"
    ]

    if os.path.exists(csv_filename):
        df_existing = pd.read_excel(csv_filename)
    else:
        df_existing = pd.DataFrame(columns=column_order)

    if sdk_name == "qiskit":
        for benchmark in tqdm(benchmarks):
            file_path = os.path.join(circuit_folder, f"{benchmark}.qasm")

            with open(file_path, "r") as f:
                qasm_str = f.read()
                qc = QuantumCircuit.from_qasm_str(qasm_str)

            transpiled_circuit = transpile(qc, basis_gates=basis_gates, optimization_level=0, seed_transpiler=0)
            num_qubits = transpiled_circuit.num_qubits

            for i, transpiler_pass in enumerate(transpiler_passes):
                start_time = time.time()
                pass_manager = PassManager(transpiler_pass)
                try:
                    optimized_circuit = pass_manager.run(transpiled_circuit)
                except Exception as e:
                    print(f"Error processing {benchmark} {transpiler_pass}")
                    print(e)
                    continue
                optimization_time = time.time() - start_time

                if transpiled_circuit != optimized_circuit:

                    original_ops = transpiled_circuit.count_ops()
                    optimized_ops = optimized_circuit.count_ops()

                    gate_count_original = sum(original_ops.values())
                    gate_count_optimized = sum(optimized_ops.values())
                    gate_count_diff = (gate_count_optimized - gate_count_original) / gate_count_original
                    
                    try:
                        qubits, runtime = estimate_resources(transpiled_circuit)
                        optimized_qubits, optimized_runtime = estimate_resources(optimized_circuit)
                    except:
                        transpiled_circuit = transpiled_circuit.remove_final_measurements()
                        optimized_circuit = optimized_circuit.remove_final_measurements()
                        qubits, runtime = estimate_resources(transpiled_circuit)
                        optimized_qubits, optimized_runtime = estimate_resources(optimized_circuit)

                    relative_qubits_delta = (optimized_qubits - qubits) / qubits
                    relative_runtime_delta = (optimized_runtime - runtime) / runtime

                    if (
                        gate_count_diff != 0 or relative_qubits_delta != 0 or 
                        relative_runtime_delta != 0
                    ):
 
                        transpiler_pass_str = str([pass_.name() for pass_ in transpiler_pass])

                        new_data = pd.DataFrame([[
                            benchmark, num_qubits, transpiler_pass_str,
                            original_ops, optimized_ops,
                            gate_count_original, gate_count_optimized,
                            qubits, optimized_qubits, runtime, optimized_runtime,
                            optimization_time
                        ]], columns=column_order)

                        df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                        df_existing.to_excel(csv_filename, index=False)
 
        
    elif sdk_name == "tket":
        
        for benchmark in tqdm(benchmarks):

            qasm_file = os.path.join(circuit_folder, benchmark + ".qasm")
            qc = circuit_from_qasm(qasm_file)

            auto_rebase_pass = AutoRebase(SINGLE_QUBIT_AND_CX_TKET_STDGATES)
            custom_rebase_pass = RebaseCustom(SINGLE_QUBIT_AND_CX_TKET_STDGATES, cx_to_cx, tk1_to_rzry)
            auto_rebase_pass.apply(qc)

            num_qubits = qc.n_qubits

            qiskit_circuit = tk_to_qiskit(qc)
            qiskit_circuit = transpile(qiskit_circuit, basis_gates=SINGLE_QUBIT_AND_CX_QISKIT_STDGATES, optimization_level=0)

            for i, transpiler_pass in enumerate(transpiler_passes):

                optimized_circuit = Circuit.from_dict(qc.to_dict())
                start_time = time.time()
                transpiler_pass.apply(optimized_circuit)
                custom_rebase_pass.apply(optimized_circuit)
                auto_rebase_pass.apply(optimized_circuit)
                optimization_time = time.time() - start_time

                if qc != optimized_circuit:
                    
                    optimized_qiskit_circuit = tk_to_qiskit(optimized_circuit)
                    optimized_qiskit_circuit = transpile(optimized_qiskit_circuit, basis_gates=SINGLE_QUBIT_AND_CX_QISKIT_STDGATES, optimization_level=0)

                    original_ops = qiskit_circuit.count_ops()
                    optimized_ops = optimized_qiskit_circuit.count_ops()
                    
                    gate_count_original = sum(original_ops.values())
                    gate_count_optimized = sum(optimized_ops.values())
                    gate_count_diff = (gate_count_optimized - gate_count_original) / gate_count_original

                    qubits, runtime = estimate_resources(qiskit_circuit)
                    optimized_qubits, optimized_runtime = estimate_resources(optimized_qiskit_circuit)

                    relative_qubits_delta = (optimized_qubits - qubits) / qubits
                    relative_runtime_delta = (optimized_runtime - runtime) / runtime
                    
                    if (
                        gate_count_diff != 0 or relative_qubits_delta != 0 or 
                        relative_runtime_delta != 0
                    ):
                        transpiler_pass_str = transpiler_passes_names[i]

                        new_data = pd.DataFrame([[
                            benchmark, num_qubits, transpiler_pass_str,
                            original_ops, optimized_ops,
                            gate_count_original, gate_count_optimized,
                            qubits, optimized_qubits, runtime, optimized_runtime,
                            optimization_time
                        ]], columns=column_order)

                        df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                        df_existing.to_excel(csv_filename, index=False)

    print(f"Results saved and updated in {csv_filename}")