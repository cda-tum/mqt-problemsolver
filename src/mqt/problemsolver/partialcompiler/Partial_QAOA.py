import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeManila, FakeMontreal, FakeWashington

P_SAMPLE_TWO_QUBIT_GATE = 0.5
IBM_GATES = ["cx", "sx", "x", "rz"]


class Partial_QAOA_Instance:
    def __init__(self, num_qubits: int, repetitions: int = 1):
        self.num_qubits = num_qubits
        self.repetitions = repetitions
        self.offline_time_edges = []
        for i in range(1, num_qubits):
            if np.random.random() < P_SAMPLE_TWO_QUBIT_GATE:
                self.offline_time_edges.append((0, i))
        self.online_time_edges = []
        self.problem_parameters: list[Parameter] = []
        self.mapping: list[int] = []
        for i in range(1, num_qubits):
            for j in range(i + 1, num_qubits):
                if np.random.random() < P_SAMPLE_TWO_QUBIT_GATE:
                    self.online_time_edges.append((i, j))

        manila_config = FakeManila().configuration()
        montreal_config = FakeMontreal().configuration()
        washington_config = FakeWashington().configuration()
        if num_qubits <= manila_config.n_qubits:
            self.coupling_map = manila_config.coupling_map
            self.device_qubits = manila_config.n_qubits
        elif num_qubits <= montreal_config.n_qubits:
            self.coupling_map = montreal_config.coupling_map
            self.device_qubits = montreal_config.n_qubits
        elif num_qubits <= washington_config.n_qubits:
            self.coupling_map = washington_config.coupling_map
            self.device_qubits = washington_config.n_qubits

    def get_online_time_edges(self) -> list[tuple[int, int]]:
        return self.online_time_edges

    def get_uncompiled_circuit_without_online_edges(
        self, include_online_edges: bool = False
    ) -> tuple[QuantumCircuit, list[QuantumCircuit], list[QuantumCircuit]]:
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

        return qc_prep, qcs_problem, qcs_mix

    def determine_mapping(self, qc: QuantumCircuit) -> list[int]:
        offline_mapped_qc = transpile(qc, coupling_map=self.coupling_map, basis_gates=IBM_GATES, optimization_level=3)

        layout = offline_mapped_qc._layout.initial_layout
        mapping = []
        for elem in layout.get_virtual_bits():
            if elem.register.name == "ancilla":
                pass
            else:
                mapping.append(layout.get_virtual_bits()[elem])
        return mapping

    def get_partially_compiled_circuit_without_online_edges(
        self,
    ) -> tuple[QuantumCircuit, list[QuantumCircuit], list[QuantumCircuit]]:
        qc_prep, qcs_problem_uncompiled, qcs_mix_uncompiled = self.get_uncompiled_circuit_without_online_edges()
        assert len(qcs_problem_uncompiled) == len(qcs_mix_uncompiled)
        qc_composed = qc_prep.copy()
        for i in range(len(qcs_problem_uncompiled)):
            qc_composed.compose(qcs_problem_uncompiled[i], inplace=True)
            qc_composed.compose(qcs_mix_uncompiled[i], inplace=True)
        self.mapping = self.determine_mapping(qc_composed)

        qc_prep_compiled = transpile(
            qc_prep,
            basis_gates=IBM_GATES,
            initial_layout=self.mapping,
            coupling_map=self.coupling_map,
            layout_method="trivial",
            optimization_level=3,
        )
        qcs_problem_compiled = []
        qcs_mixer_compiled = []
        for i in range(len(qcs_problem_uncompiled)):
            qcs_problem_compiled.append(
                transpile(
                    qcs_problem_uncompiled[i],
                    basis_gates=IBM_GATES,
                    initial_layout=self.mapping,
                    coupling_map=self.coupling_map,
                    layout_method="trivial",
                    optimization_level=3,
                )
            )

            qcs_mixer_compiled.append(
                transpile(
                    qcs_mix_uncompiled[i],
                    basis_gates=IBM_GATES,
                    initial_layout=self.mapping,
                    coupling_map=self.coupling_map,
                    layout_method="trivial",
                    optimization_level=3,
                )
            )

        return qc_prep_compiled, qcs_problem_compiled, qcs_mixer_compiled

    def get_uncompiled_fully_composed_circuit(self) -> QuantumCircuit:
        qc_prep, qcs_problem, qcs_mix = self.get_uncompiled_circuit_without_online_edges(include_online_edges=True)
        qc = qc_prep
        for i in range(len(qcs_problem)):
            qc.compose(qcs_problem[i], inplace=True)
            qc.compose(qcs_mix[i], inplace=True)
        return qc

    def compile_full_circuit(self) -> QuantumCircuit:
        qc = self.get_uncompiled_fully_composed_circuit()
        return transpile(
            qc,
            coupling_map=self.coupling_map,
            basis_gates=IBM_GATES,
            optimization_level=1,
        )

    def get_uncompiled_online_edges(self) -> list[QuantumCircuit]:
        qc_online_edges_uncompiled_all_reps = []
        for i in range(self.repetitions):
            qc_online_edges = QuantumCircuit(self.num_qubits)
            for elem in self.online_time_edges:
                qc_online_edges.rzz(self.problem_parameters[i], elem[0], elem[1])
            qc_online_edges_uncompiled_all_reps.append(qc_online_edges)

        return qc_online_edges_uncompiled_all_reps

    def get_compiled_online_edges(self) -> list[QuantumCircuit]:
        assert self.mapping
        qc_online_edges_compiled_all_reps = []
        qc_online_edges_uncompiled_all_reps = self.get_uncompiled_online_edges()
        for qc in qc_online_edges_uncompiled_all_reps:
            qc_online_edges_compiled_all_reps.append(
                transpile(
                    qc,
                    basis_gates=IBM_GATES,
                    initial_layout=self.mapping,
                    layout_method="trivial",
                    coupling_map=self.coupling_map,
                    optimization_level=1,
                )
            )

        return qc_online_edges_compiled_all_reps
