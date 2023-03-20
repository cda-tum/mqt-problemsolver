from __future__ import annotations

from typing import Any

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
        """
        Creates a QAOA problem instance with a random number of known/offline edges and a random number of unknown/online edges.
        :param num_qubits: Number of qubits in the problem instance
        :param repetitions: Number of repetitions of the problem and mixer unitaries
        """
        self.num_qubits = num_qubits
        self.repetitions = repetitions
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

        qc, qc_baseline = self.get_uncompiled_circuits()
        self.qc = qc
        self.qc_baseline = qc_baseline
        self.qc_compiled = self.compile_qc(baseline=False, opt_level=2)
        self.to_be_checked_gates_indices = self.get_to_be_checked_gates()

    def get_uncompiled_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        qc = QuantumCircuit(self.num_qubits)
        qc_baseline = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        qc_baseline.h(range(self.num_qubits))

        self.remove_gates = []
        for k in range(self.repetitions):
            p = Parameter(f"a_{k}")
            for i in range(self.num_qubits):
                for j in range(i + 1, min(self.num_qubits, i + 3)):
                    qc.rzz(p, i, j)
                    if np.random.random() < self.sample_probability:
                        self.remove_gates.append(True)
                    else:
                        self.remove_gates.append(False)
                        qc_baseline.rzz(p, i, j)

            m = Parameter(f"b_{k}")

            qc.rx(2 * m, range(self.num_qubits))
            qc_baseline.rx(2 * m, range(self.num_qubits))

        return qc, qc_baseline

    def compile_qc(self, baseline: bool = False, opt_level: int = 2) -> QuantumCircuit:
        if baseline:
            return transpile(self.qc_baseline, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)
        return transpile(self.qc, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)

    def get_to_be_checked_gates(self) -> list[int]:
        indices = []
        for i, gate in enumerate(self.qc_compiled._data):
            if (
                gate.operation.name == "rz"
                and isinstance(gate.operation.params[0], Parameter)
                and gate.operation.params[0].name.startswith("a_")
            ):
                indices.append(i)
        return indices


def reduce_swaps(qc: QuantumCircuit) -> QuantumCircuit:
    transpile_passes = [
        CommutativeCancellation(),
        CommutativeInverseCancellation(),
        CXCancellation(),
        Collect2qBlocks(),
        ConsolidateBlocks(),
    ]
    return PassManager(transpile_passes).run(qc).decompose()


def check_gates(qc: QuantumCircuit, remove_gates: list[bool], to_be_checked_gates_indices: list[Any]) -> QuantumCircuit:
    # assert len(to_be_checked_gates_indices) == len(remove_gates)
    offset = 0
    for i in range(len(remove_gates)):
        # assert isinstance(to_be_checked_gates_indices[i].operation.params[0], Parameter)
        if remove_gates[i]:
            del qc._data[to_be_checked_gates_indices[i - offset]]
            offset += 1
        # qc._data.remove(to_be_checked_gates_indices[i])
    return qc
