from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeManila
import numpy as np

class Partial_QAOA_Instance:
    def __init__(self, num_qubits:int, repetitions:int):

        self.num_qubits = num_qubits
        self.repetitions = repetitions
        self.known_vertices = []
        for i in range(num_qubits):
            any_interaction = False
            for j in range(i, num_qubits):
                if np.random()>0.5:
                    any_interaction = True
                    self.known_vertices.append((i, j))
            if not any_interaction:
                if i > 0:
                    self.known_vertices.append((i, i-1))
                else:
                    self.known_vertices.append((i, i+1))




    def get_qaoa_uncompiled(self) -> tuple[QuantumCircuit, list[QuantumCircuit], [QuantumCircuit]]:
        qc_prep = QuantumCircuit(self.num_qubits)
        qc_prep.h(range(self.num_qubits))
        qc_prep.barrier()
        qcs_problem = []
        qcs_mix = []
        qc_problem = QuantumCircuit(self.num_qubits)
        for i in range(self.repetitions):
            p = Parameter(f"a_{i}")
            for elem in self.known_vertices:
                qc_problem.rzz(p, elem[0], elem[1])
            qc_problem.barrier()
            qcs_problem.append(qc_problem)

            m = Parameter(f"b_{i}")
            qc_mix = QuantumCircuit(self.num_qubits)
            qc_mix.rx(2 * m, range(self.num_qubits))
            qc_mix.barrier()
            qcs_mix.append(qc_mix)

        return qc_prep, qcs_problem, qcs_mix


    def get_mapping(self, qc: QuantumCircuit) -> list[int]:
        coupling_map = FakeManila().configuration().coupling_map
        offline_mapped_qc = transpile(
            qc, coupling_map=coupling_map, basis_gates=["cx", "sx", "x", "rz"], optimization_level=3
        )

        layout = offline_mapped_qc._layout.initial_layout
        mapping = []
        for elem in layout.get_virtual_bits():
            if elem.register.name == "ancilla":
                pass
            else:
                mapping.append(layout.get_virtual_bits()[elem])

        return mapping


    def get_qaoa_partially_compiled(self) -> tuple[QuantumCircuit, list[QuantumCircuit], [QuantumCircuit]]:
        qc_prep, qcs_problem_uncompiled, qcs_mix_uncompiled = self.get_qaoa_uncompiled()
        assert len(qcs_problem_uncompiled) == len(qcs_mix_uncompiled)
        qc_composed = qc_prep.copy()
        for i in range(len(qcs_problem_uncompiled)):
            qc_composed.compose(qcs_problem_uncompiled[i], inplace=True)
            qc_composed.compose(qcs_mix_uncompiled[i], inplace=True)
        mapping = self.get_mapping(qc_composed)

        qc_prep_compiled = transpile(qc_prep,
                basis_gates=["cx", "sx", "x", "rz"],
                initial_layout=mapping,
                layout_method="trivial",
                optimization_level=3,
            )
        qcs_problem_compiled = []
        qcs_mixer_compiled = []
        for i in range(len(qcs_problem_uncompiled)):
            qcs_problem_compiled.append(transpile(
                qcs_problem_uncompiled[i],
                basis_gates=["cx", "sx", "x", "rz"],
                initial_layout=mapping,
                layout_method="trivial",
                optimization_level=3,
            ))

            qcs_mixer_compiled.append(transpile(
                qcs_mix_uncompiled,
                basis_gates=["cx", "sx", "x", "rz"],
                initial_layout=mapping,
                layout_method="trivial",
                optimization_level=3,
            ))

        return qc_prep_compiled, qcs_problem_compiled, qcs_mixer_compiled
