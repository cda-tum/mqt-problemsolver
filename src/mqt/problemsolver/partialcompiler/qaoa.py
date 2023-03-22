from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeManila, FakeMontreal, FakeWashington
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    CXCancellation,
)


class QAOA:
    def __init__(self, num_qubits: int, repetitions: int = 1, sample_probability: float = 0.5):
        self.num_qubits = num_qubits
        self.repetitions = repetitions
        assert 0 <= sample_probability <= 1
        self.sample_probability = sample_probability
        np.random.seed(42)
        manila_config = FakeManila().configuration()
        montreal_config = FakeMontreal().configuration()
        washington_config = FakeWashington().configuration()
        if num_qubits <= manila_config.n_qubits:
            self.backend = FakeManila()
        elif num_qubits <= montreal_config.n_qubits:
            self.backend = FakeMontreal()
        elif num_qubits <= washington_config.n_qubits:
            self.backend = FakeWashington()

        qc, qc_baseline, remove_gates = self.get_uncompiled_circuits()
        self.qc = qc  # QC with all gates
        self.qc_baseline = qc_baseline  # QC with only the sampled gates
        self.remove_gates = remove_gates  # List of booleans indicating whether a gate should be removed
        self.qc_compiled = self.compile_qc(baseline=False, opt_level=2)  # Compiled QC with all gates
        self.to_be_checked_gates_indices = self.get_to_be_removed_gate_indices()  # Indices of the gates to be checked

    def get_uncompiled_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit, list[bool]]:
        qc = QuantumCircuit(self.num_qubits)
        qc_baseline = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        qc_baseline.h(range(self.num_qubits))

        remove_gates = []
        parameter_counter = 0
        for k in range(self.repetitions):
            for i in range(self.num_qubits):
                for j in range(i + 1, min(self.num_qubits, i + 3)):
                    p = Parameter(f"a_{parameter_counter}")
                    qc.rzz(p, i, j)
                    if np.random.random() < self.sample_probability:
                        remove_gates.append(p.name)
                    else:
                        remove_gates.append(False)
                        qc_baseline.rzz(p, i, j)
                    parameter_counter += 1

            m = Parameter(f"b_{k}")
            qc.rx(2 * m, range(self.num_qubits))
            qc_baseline.rx(2 * m, range(self.num_qubits))

        qc.measure_all()
        qc_baseline.measure_all()

        return qc, qc_baseline, remove_gates

    def compile_qc(self, baseline: bool = False, opt_level: int = 2) -> QuantumCircuit:
        if baseline:
            return transpile(self.qc_baseline, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)
        return transpile(self.qc, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)

    def get_to_be_removed_gate_indices(self) -> list[int]:
        indices_parameterized_gates = []
        for i, gate in enumerate(self.qc_compiled._data):
            if (
                gate.operation.name == "rz"
                and isinstance(gate.operation.params[0], Parameter)
                and gate.operation.params[0].name.startswith("a_")
            ):
                indices_parameterized_gates.append(i)

        assert len(indices_parameterized_gates) == len(self.remove_gates)
        indices_to_be_removed_parameterized_gates = []
        for i in range(len(indices_parameterized_gates)):
            if self.qc_compiled._data[indices_parameterized_gates[i]].operation.params[0].name in self.remove_gates:
                indices_to_be_removed_parameterized_gates.append(indices_parameterized_gates[i])

        assert len(set(indices_to_be_removed_parameterized_gates)) == len({elem for elem in self.remove_gates if elem})
        return indices_to_be_removed_parameterized_gates

    def check_gates(self, qc: QuantumCircuit, optimize_swaps: bool = True) -> QuantumCircuit:
        offset = 0
        counter = 0
        for i in self.to_be_checked_gates_indices:
            # Remove the gate from the ParameterTable
            # del qc._parameter_table[qc._data[i - offset].operation.params[0]]._instance_ids[
            #     (id(qc._data[i - offset].operation), 0)
            # ]
            del qc._parameter_table[
                qc._data[i - offset].operation.params[0]
            ]  # remove the Parameter from the ParameterTable for the specific parameter. The table just contains one entry
            if (
                optimize_swaps
                and qc._data[i - offset - 1].operation.name == "cx"
                and qc._data[i - offset - 1] == qc._data[i - offset + 1]
            ):
                del qc._data[i - offset - 1]
                del qc._data[i - offset - 1]
                del qc._data[i - offset - 1]

                offset += 3
            else:
                del qc._data[i - offset]
                offset += 1
            counter += 1

        return qc


def reduce_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    transpile_passes = [
        CommutativeCancellation(),
        CommutativeInverseCancellation(),
        CXCancellation(),
        Collect2qBlocks(),
        ConsolidateBlocks(),
    ]
    return PassManager(transpile_passes).run(qc).decompose()