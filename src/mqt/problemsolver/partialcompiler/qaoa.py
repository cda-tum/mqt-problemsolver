from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.providers import BackendV2

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import GenericBackendV2


class QAOA:
    def __init__(
        self,
        num_qubits: int,
        repetitions: int = 1,
        sample_probability: float = 0.5,
        considered_following_qubits: int = 3,
        satellite_use_case: bool = False,
    ):
        self.num_qubits = num_qubits
        self.repetitions = repetitions

        assert 0 <= sample_probability <= 1
        self.sample_probability = sample_probability

        self.backend = get_backend(num_qubits)
        self.satellite_use_case = satellite_use_case
        qc, qc_baseline, remove_gates, remove_pairs = self.get_uncompiled_circuits(considered_following_qubits)
        self.qc = qc  # QC with all gates
        self.qc_baseline = qc_baseline  # QC with only the sampled gates
        self.remove_pairs = remove_pairs  # List of all the to be removed ZZ gates between qubit pairs
        self.remove_gates = remove_gates  # List of length number of parameterized gates, contains either False (if it shall not be removed) or the parameter name of the gate to be removed
        self.qc_compiled = self.compile_qc(baseline=False, opt_level=3)  # Compiled QC with all gates
        self.to_be_removed_gates_indices = self.get_to_be_removed_gate_indices()  # Indices of the gates to be checked
        self.penalty = 5  # Penalty for the QUBO model

    def get_uncompiled_circuits(
        self,
        considered_following_qubits: int,
    ) -> tuple[QuantumCircuit, QuantumCircuit, list[bool | str], list[tuple[int, int]]]:
        """Returns the uncompiled circuits (both with only the actual needed two-qubit gates and with all possible
        two-qubit gates) and the list of gates to be removed."""

        qc = QuantumCircuit(self.num_qubits)  # QC with all gates
        qc_baseline = QuantumCircuit(self.num_qubits)  # QC with only the sampled gates
        qc.h(range(self.num_qubits))
        qc_baseline.h(range(self.num_qubits))

        remove_gates: list[bool | str] = []
        parameter_counter = 0
        tmp_len = -1
        rng = np.random.default_rng(seed=42)

        remove_pairs = []
        # Iterate over all QAOA layers
        for k in range(self.repetitions):
            # for the satellite use case, rz gates are added which represent the location image value. These factors are set later on when all interactions are known.
            if self.satellite_use_case:
                for i in range(self.num_qubits):
                    p_qubit = Parameter(f"qubit_{i}_rep_{k}")
                    qc.rz(2 * p_qubit, i)
                    qc_baseline.rz(2 * p_qubit, i)
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
                        if rng.random() < (1 - self.sample_probability):
                            remove_gates.append(p.name)
                        else:
                            remove_gates.append(False)
                            qc_baseline.rzz(p, i, j)
                            remove_pairs.append((i, j))
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

        return qc, qc_baseline, remove_gates, remove_pairs

    def compile_qc(self, baseline: bool = False, opt_level: int = 3) -> QuantumCircuit:
        """Compiles the circuit"""
        circ = self.qc_baseline if baseline else self.qc
        assert self.backend is not None
        qc_comp = transpile(circ, backend=self.backend, optimization_level=opt_level, seed_transpiler=42)
        if baseline and self.satellite_use_case:
            return self.apply_factors_to_qc(qc_comp)
        return qc_comp

    def get_to_be_removed_gate_indices(self) -> list[int]:
        """Returns the indices of the gates to be removed"""
        indices_to_be_removed_parameterized_gates = []
        for i, gate in enumerate(self.qc_compiled._data):
            if (
                gate.operation.name == "rz"
                and isinstance(gate.operation.params[0], Parameter)
                and gate.operation.params[0].name.startswith("a_")
            ) and gate.operation.params[0].name in self.remove_gates:
                indices_to_be_removed_parameterized_gates.append(i)
        assert len(set(indices_to_be_removed_parameterized_gates)) == len({elem for elem in self.remove_gates if elem})
        return indices_to_be_removed_parameterized_gates

    def remove_unnecessary_gates(self, qc: QuantumCircuit, optimize_swaps: bool = True) -> QuantumCircuit:
        """Removes the gates to be checked from the circuit at online time"""
        indices = set()

        # Iterate over all gates to be removed
        for i in self.to_be_removed_gates_indices:
            # Remove the Parameter from the ParameterTable for the specific parameter.
            param = qc._data[i].operation.params[0]
            if param in qc.parameters:
                qc.parameters.remove(param)
            indices.add(i)
            if optimize_swaps and qc._data[i - 1].operation.name == "cx" and qc._data[i - 1] == qc._data[i + 1]:
                indices.add(i - 1)
                indices.add(i + 1)

        qc._data = [v for i, v in enumerate(qc._data) if i not in indices]

        if self.satellite_use_case:
            return self.apply_factors_to_qc(qc)

        return qc

    def create_model_from_pair_list(self) -> NDArray[np.float64]:
        """
        Constructs the QUBO matrix Q for the optimization problem.

        The matrix Q is of size num_qubits x num_qubits and encodes the following:
          - Objective: Minimize the linear term -∑ x_i, where Q[i, i] = -1.
          - Constraints: Ensure x_i + x_j ≤ 1 by adding a penalty term P·x_i x_j
            for each pair (i, j) in remove_pairs.

        Returns:
            Q: The QUBO matrix representing the optimization problem.
        """
        n = self.num_qubits
        Q = np.zeros((n, n))
        # linear objective: minimize -∑ x_i
        for i in range(n):
            Q[i, i] = -1.0

        # conflict penalties
        for i, j in self.remove_pairs:
            Q[i, j] += self.penalty
            Q[j, i] += self.penalty

        return Q

    def apply_factors_to_qc(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Updates the parameterized QAOA-style circuit `qc` with actual Ising coefficients.

        This function assigns values to the circuit parameters based on the Ising model
        derived from the QUBO matrix. Parameters are named "qubit_<i>" for single-qubit
        Z-rotations and "a_<i>_<j>" (or just "a_...") for ZZ-interaction gates.

        Args:
            qc: QuantumCircuit object representing the parameterized QAOA circuit.

        Returns:
            QuantumCircuit object with updated parameter values based on the Ising model.
        """
        # 1) build QUBO
        Q = self.create_model_from_pair_list()

        # 2) map to Ising: H = x^T Q x  with  x = (1-Z)/2
        #    ⇒ h_i = -½ ∑_j Q[i,j] ,    J_{ij} = ½ Q[i,j]  (i≠j)
        h = -np.sum(Q, axis=1) / 2.0
        J = Q.copy() / 2.0
        np.fill_diagonal(J, 0.0)

        # 3) assign each qc parameter to 2y·h or 2y·J  (the factor of 2 comes
        #    from the fact that Rz(θ) = exp(-i θ/2 Z))
        for param in qc.parameters:
            name = param.name

            if name.startswith("qubit_"):
                # single-qubit Z-term
                idx = int(name.split("_", 1)[1])
                qc.assign_parameters({param: 2 * h[idx] * param}, inplace=True)

            elif name.startswith("a_"):
                parts = name.split("_")
                if len(parts) == 3:
                    # per-edge interaction
                    i, j = map(int, parts[1:])
                    qc.assign_parameters({param: 2 * J[i, j] * param}, inplace=True)
                else:
                    # "global" a-parameter: sum all J's
                    total_J = J[np.triu_indices(self.num_qubits, 1)].sum()
                    qc.assign_parameters({param: 2 * total_J * param}, inplace=True)

        return qc


def get_backend(num_qubits: int) -> BackendV2:
    return GenericBackendV2(num_qubits=num_qubits, noise_info=True)
