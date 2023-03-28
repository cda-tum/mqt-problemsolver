from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeBackend, FakeManila, FakeMontreal, FakeWashington


class QAOA:
    def __init__(
        self,
        num_qubits: int,
        repetitions: int = 1,
        remove_probability: float = 0.5,
        considered_following_qubits: int = 3,
    ):
        self.num_qubits = num_qubits
        self.repetitions = repetitions

        assert 0 <= remove_probability <= 1
        self.sample_probability = remove_probability
        np.random.seed(42)

        self.backend = get_backend(num_qubits)

        qc, qc_baseline, remove_gates = self.get_uncompiled_circuits(considered_following_qubits)
        self.qc = qc  # QC with all gates
        self.qc_baseline = qc_baseline  # QC with only the sampled gates
        self.remove_gates = remove_gates  # List of booleans indicating whether a gate should be removed
        self.qc_compiled = self.compile_qc(baseline=False, opt_level=3)  # Compiled QC with all gates
        self.to_be_removed_gates_indices = self.get_to_be_removed_gate_indices()  # Indices of the gates to be checked

    def get_uncompiled_circuits(
        self, considered_following_qubits: int
    ) -> tuple[QuantumCircuit, QuantumCircuit, list[bool | str]]:
        """Returns the uncompiled circuits (both with only the actual needed two-qubit gates and with all possible
        two-qubit gates) and the list of gates to be removed."""

        qc = QuantumCircuit(self.num_qubits)  # QC with all gates
        qc_baseline = QuantumCircuit(self.num_qubits)  # QC with only the sampled gates
        qc.h(range(self.num_qubits))
        qc_baseline.h(range(self.num_qubits))

        remove_gates: list[bool | str] = []
        parameter_counter = 0
        tmp_len = -1

        # Iterate over all QAOA layers
        for k in range(self.repetitions):
            if k == 1:
                tmp_len = len(remove_gates)  # Number of parameterized gates in the first layer

            # Iterate over all possible two-qubit gates
            for i in range(self.num_qubits):
                # considered_following_qubits describes the number of to be considered following qubits
                for j in range(i + 1, i + 1 + considered_following_qubits):
                    if j >= self.num_qubits:
                        break
                    p = Parameter(f"a_{parameter_counter}")
                    qc.rzz(p, i, j)
                    # Sample whether the gate should be removed for the first layer
                    if k == 0:
                        if np.random.random() < self.sample_probability:
                            remove_gates.append(p.name)
                        else:
                            remove_gates.append(False)
                            qc_baseline.rzz(p, i, j)
                    # For all other layers, check whether the gate should be removed
                    elif remove_gates[parameter_counter - k * tmp_len]:
                        remove_gates.append(p.name)
                    else:
                        remove_gates.append(False)
                        qc_baseline.rzz(p, i, j)
                    parameter_counter += 1

            m = Parameter(f"b_{k}")

            # Mixer Layer
            qc.rx(2 * m, range(self.num_qubits))
            qc_baseline.rx(2 * m, range(self.num_qubits))

        qc.measure_all()
        qc_baseline.measure_all()

        return qc, qc_baseline, remove_gates

    def compile_qc(self, baseline: bool = False, opt_level: int = 3) -> QuantumCircuit:
        """Compiles the circuit"""
        if baseline:
            return transpile(self.qc_baseline, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)
        return transpile(self.qc, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)

    def get_to_be_removed_gate_indices(self) -> list[int]:
        """Returns the indices of the gates to be removed"""
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
        """Removes the gates to be checked from the circuit at online time"""
        offset = 0

        # Iterate over all gates to be removed
        for i in self.to_be_removed_gates_indices:
            # Remove the gate from the ParameterTable (was needed for QCEC to work)
            del qc._parameter_table[
                qc._data[i - offset].operation.params[0]
            ]  # remove the Parameter from the ParameterTable for the specific parameter. The table just contains one entry
            # Delete the gate and optionally the enclosing CX gates
            if (
                optimize_swaps
                and qc._data[i - offset - 1].operation.name == "cx"
                and qc._data[i - offset - 1] == qc._data[i - offset + 1]
            ):
                del qc._data[i - offset - 1 : i - offset + 2]

                offset += 3
            else:
                del qc._data[i - offset]
                offset += 1

        return qc


def get_backend(num_qubits: int) -> FakeBackend:
    FakeManila().configuration()
    montreal_config = FakeMontreal().configuration()
    washington_config = FakeWashington().configuration()
    manila_config = FakeManila().configuration()
    FakeManila().configuration()
    if num_qubits <= manila_config.n_qubits:
        return FakeManila()
    if num_qubits <= montreal_config.n_qubits:
        return FakeMontreal()
    if num_qubits <= washington_config.n_qubits:
        return FakeWashington()
    return None
