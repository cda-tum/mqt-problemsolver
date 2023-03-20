from __future__ import annotations

from typing import Literal, overload

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeManila, FakeMontreal, FakeWashington

P_SAMPLE_TWO_QUBIT_GATE = 0.5


class Partial_QAOA:
    def __init__(self, num_qubits: int, repetitions: int = 1):
        """
        Creates a QAOA problem instance with a random number of known/offline edges and a random number of unknown/online edges.
        :param num_qubits: Number of qubits in the problem instance
        :param repetitions: Number of repetitions of the problem and mixer unitaries
        """
        self.num_qubits = num_qubits
        self.repetitions = repetitions

        manila_config = FakeManila().configuration()
        montreal_config = FakeMontreal().configuration()
        washington_config = FakeWashington().configuration()
        if num_qubits <= manila_config.n_qubits:
            self.backend = FakeManila()
        elif num_qubits <= montreal_config.n_qubits:
            self.backend = FakeMontreal()
        elif num_qubits <= washington_config.n_qubits:
            self.backend = FakeWashington()


    def get_uncompiled_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit]:

        qc = QuantumCircuit(self.num_qubits)
        qc_baseline = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        qc_baseline.h(range(self.num_qubits))
        qc.barrier()
        qc_baseline.barrier()
        self.problem_parameters = []
        self.remove_gates = []

        for i in range(self.repetitions):
            p = Parameter(f"a_{i}")
            self.problem_parameters.append(p)
            for i in range(self.num_qubits):
                for j in range(i + 1, min(self.num_qubits, i + 3)):
                    qc.rzz(p, i, j)
                    if np.random.random() < P_SAMPLE_TWO_QUBIT_GATE:
                        self.remove_gates.append(True)
                    else:
                        self.remove_gates.append(False)
                        qc_baseline.rzz(p, i, j)

            m = Parameter(f"b_{i}")
            qc.barrier()
            qc_baseline.barrier()
            qc.rx(2 * m, range(self.num_qubits))
            qc_baseline.rx(2 * m, range(self.num_qubits))
            qc.barrier()
            qc_baseline.barrier()

        return qc, qc_baseline


