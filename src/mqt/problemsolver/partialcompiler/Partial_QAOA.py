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
        self.offline_time_edges = []

        # add offline edges
        for i in range(1, num_qubits):
            if np.random.random() < P_SAMPLE_TWO_QUBIT_GATE:
                self.offline_time_edges.append((0, i))
        self.problem_parameters: list[Parameter] = []
        self.mapping: list[int] = []

        # add online edges
        self.online_time_edges = []
        for i in range(1, num_qubits):
            for j in range(i + 1, num_qubits):
                if np.random.random() < P_SAMPLE_TWO_QUBIT_GATE:
                    self.online_time_edges.append((i, j))

        # determine device
        manila_config = FakeManila().configuration()
        montreal_config = FakeMontreal().configuration()
        washington_config = FakeWashington().configuration()
        if num_qubits <= manila_config.n_qubits:
            self.backend = FakeManila()
        elif num_qubits <= montreal_config.n_qubits:
            self.backend = FakeMontreal()
        elif num_qubits <= washington_config.n_qubits:
            self.backend = FakeWashington()

        self.mapping = self.get_predetermined_mapping()

    @overload
    def get_uncompiled_circuit(
        self, return_as_one_circuit: Literal[True], include_online_edges: bool = False
    ) -> QuantumCircuit:
        ...

    @overload
    def get_uncompiled_circuit(
        self,
        return_as_one_circuit: Literal[False],
        include_online_edges: bool = False,
    ) -> tuple[QuantumCircuit, list[QuantumCircuit], list[QuantumCircuit]]:
        ...

    def get_uncompiled_circuit(
        self, return_as_one_circuit: bool = False, include_online_edges: bool = False
    ) -> tuple[QuantumCircuit, list[QuantumCircuit], list[QuantumCircuit]] | QuantumCircuit:
        """Return the state preparation circuit and lists of problem and mixer circuits without any compilation.
         Online edges are optionally included.

        :param include_online_edges: Include online edges in the problem circuits
        """
        qc_prep = QuantumCircuit(self.num_qubits)
        qc_prep.h(range(self.num_qubits))
        qc_prep.barrier()
        qcs_problem = []
        qcs_mix = []
        self.problem_parameters = []
        for i in range(self.repetitions):
            qc_problem = QuantumCircuit(self.num_qubits)
            p = Parameter(f"a_{i}")
            self.problem_parameters.append(p)
            for elem in self.offline_time_edges:
                qc_problem.rzz(p, elem[0], elem[1])
            if include_online_edges:
                for elem in self.online_time_edges:
                    qc_problem.rzz(p, elem[0], elem[1])

            qcs_problem.append(qc_problem)

            m = Parameter(f"b_{i}")
            qc_mix = QuantumCircuit(self.num_qubits)
            qc_mix.barrier()
            qc_mix.rx(2 * m, range(self.num_qubits))
            qc_mix.barrier()
            qcs_mix.append(qc_mix)
        if return_as_one_circuit:
            qc_composed = qc_prep
            for i in range(len(qcs_problem)):
                qc_composed.compose(qcs_problem[i], inplace=True)
                qc_composed.compose(qcs_mix[i], inplace=True)
            return qc_composed
        return qc_prep, qcs_problem, qcs_mix

    def get_predetermined_mapping(self) -> list[int]:
        """Determines a mapping layout to the selected quantum devices based on the offline edges."""
        qc = self.get_uncompiled_circuit(return_as_one_circuit=True, include_online_edges=False)
        offline_mapped_qc = transpile(
            qc,
            coupling_map=self.backend.configuration().coupling_map,
            basis_gates=self.backend.configuration().basis_gates,
            optimization_level=3,
        )

        layout = offline_mapped_qc._layout.initial_layout
        mapping = []
        for elem in layout.get_virtual_bits():
            if elem.register.name == "ancilla":
                pass
            else:
                mapping.append(layout.get_virtual_bits()[elem])

        return mapping

    def get_partially_compiled_circuit_without_online_edges(
        self, consider_mapping: bool = True
    ) -> tuple[QuantumCircuit, list[QuantumCircuit], list[QuantumCircuit]]:
        """
        Return the state preparation circuit and lists of problem and mixer circuits with the option to consider
        the mapping with all returned circuits are compiled to native gates and optionally mapped to the device.
        """

        qc_prep, qcs_problem_uncompiled, qcs_mix_uncompiled = self.get_uncompiled_circuit(return_as_one_circuit=False)
        assert isinstance(qcs_problem_uncompiled, list)
        assert isinstance(qcs_mix_uncompiled, list)

        # mapping_fct = self.compile_with_mapping if consider_mapping else self.compile_without_mapping
        if consider_mapping:
            qc_prep_compiled = self.compile_with_mapping(qc_prep)
            qcs_problem_compiled = [self.compile_with_mapping(qc) for qc in qcs_problem_uncompiled]
            qcs_mixer_compiled = [self.compile_with_mapping(qc) for qc in qcs_mix_uncompiled]
        else:
            qc_prep_compiled = self.compile_without_mapping(qc_prep)
            qcs_problem_compiled = [self.compile_without_mapping(qc) for qc in qcs_problem_uncompiled]
            qcs_mixer_compiled = [self.compile_without_mapping(qc) for qc in qcs_mix_uncompiled]

        if consider_mapping:
            problem_compiled_final_layout = qcs_problem_compiled[0]._layout.final_layout
            if problem_compiled_final_layout is not None:
                lut = problem_compiled_final_layout.get_physical_bits()
                for elem in self.mapping:
                    if elem != lut[elem].index:
                        print(elem, "->", lut[elem].index)

        return qc_prep_compiled, qcs_problem_compiled, qcs_mixer_compiled

    def compile_without_mapping(self, qc: QuantumCircuit, opt_level: int = 3) -> QuantumCircuit:
        return transpile(
            qc,
            basis_gates=self.backend.configuration().basis_gates,
            optimization_level=opt_level,
        )

    def compile_with_mapping(self, qc: QuantumCircuit, opt_level: int = 0) -> QuantumCircuit:
        return transpile(
            qc,
            basis_gates=self.backend.configuration().basis_gates,
            initial_layout=self.mapping,
            coupling_map=self.backend.configuration().coupling_map,
            optimization_level=opt_level,
        )

    def compile_full_circuit(self) -> QuantumCircuit:
        """
        Returns the fully composed circuit compiled to the selected quantum device.
        """
        qc = self.get_uncompiled_circuit(return_as_one_circuit=True, include_online_edges=True)
        return transpile(
            qc,
            coupling_map=self.backend.configuration().coupling_map,
            basis_gates=self.backend.configuration().basis_gates,
            optimization_level=1,
        )

    def get_uncompiled_online_edges(self) -> list[QuantumCircuit]:
        """
        Returns the online edges uncompiled for all repetitions.
        """
        qc_online_edges_uncompiled_all_reps = [QuantumCircuit(self.num_qubits) for _ in range(self.repetitions)]
        for i, qc in enumerate(qc_online_edges_uncompiled_all_reps):
            for q1, q2 in self.online_time_edges:
                qc.rzz(self.problem_parameters[i], q1, q2)

        return qc_online_edges_uncompiled_all_reps

    def get_compiled_online_edges(self) -> list[QuantumCircuit]:
        """
        Returns the online edges compiled for all repetitions.
        """
        assert self.mapping
        qc_online_edges_uncompiled_all_reps = self.get_uncompiled_online_edges()

        return [
            self.compile_with_mapping(qc, opt_level=1, layout_method="trivial")
            for qc in qc_online_edges_uncompiled_all_reps
        ]
