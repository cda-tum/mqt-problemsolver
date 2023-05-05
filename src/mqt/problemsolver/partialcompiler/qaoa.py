from __future__ import annotations

import numpy as np
from docplex.mp.model import Model
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeBackend, FakeMontreal, FakeQuito, FakeWashington
from qiskit_optimization.converters.quadratic_program_to_qubo import (
    QuadraticProgramToQubo,
)
from qiskit_optimization.translators import from_docplex_mp


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
            del qc._parameter_table[qc._data[i].operation.params[0]]
            indices.add(i)
            if optimize_swaps and qc._data[i - 1].operation.name == "cx" and qc._data[i - 1] == qc._data[i + 1]:
                indices.add(i - 1)
                indices.add(i + 1)

        qc._data = [v for i, v in enumerate(qc._data) if i not in indices]

        if self.satellite_use_case:
            return self.apply_factors_to_qc(qc)

        return qc

    def apply_factors_to_qc(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Applies factors to each qubit representing the location image value and the dependencies to other image locations."""
        # create QUBO formulation based on interactions of between qubits
        qubo = QuadraticProgramToQubo().convert(from_docplex_mp(self.create_model_from_pair_list()))
        # extract the factors: one for each qubit/image location, one factor for all ZZ interactions, one factor for all mixer layers
        ising = qubo.to_ising()
        coeffs = np.array(ising[0].primitive.coeffs, dtype=float)
        coeffs_qubits = coeffs[: self.num_qubits]
        coeffs_interactions = coeffs[self.num_qubits + 1]

        # apply the factors, i.e. multiply the parameters with the factors
        for param in qc.parameters:
            if "a_" in param.name:
                # factor of 2 is applied to the problem layer gates since this was not done at initialization
                # (in comparison to the mixer layer), because otherwise the removal of the gates would be more complicated
                # since the gates would have ParameterExpression and not Parameter objects
                qc.assign_parameters({param: coeffs_interactions * param * 2}, inplace=True)
            elif "qubit_" in param.name:
                qc.assign_parameters({param: coeffs_qubits[int(param.name.split("_")[1])] * param}, inplace=True)

        return qc

    def create_model_from_pair_list(self) -> Model:
        """Creates a model from the interaction pairs"""
        mdl = Model("satellite model")
        locations = mdl.binary_var_list(self.num_qubits, name="locations")
        for i, j in self.remove_pairs:
            mdl.add_constraint((locations[i] + locations[j]) <= 1)
        mdl.minimize(-mdl.sum(locations[i] for i in range(self.num_qubits)))
        return mdl


def get_backend(num_qubits: int) -> FakeBackend:
    quito = FakeQuito()
    if num_qubits <= quito.configuration().n_qubits:
        return quito

    montreal = FakeMontreal()
    if num_qubits <= montreal.configuration().n_qubits:
        return montreal

    washington = FakeWashington()
    if num_qubits <= washington.configuration().n_qubits:
        return washington

    return None
