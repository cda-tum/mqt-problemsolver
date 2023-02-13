import networkx as nx
import numpy as np
from qiskit_optimization.applications import Maxcut
from qiskit import QuantumCircuit
from qiskit import transpile

from qiskit.providers.fake_provider import FakeManila
from qiskit.circuit import Parameter

def create_maxcut_quadratic_problem(num_nodes:int):
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, num_nodes, 1))
    elist = [(0, 1, 1.0), (0, 2, 1.0), (2, 3, 1.0), (1, 2, 1.0)]  # ,(0,2,1.0)
    # tuple is (i,j,weight) where (i,j) is the edge
    G.add_weighted_edges_from(elist)
    colors = ['r' for node in G.nodes()]
    pos = nx.spring_layout(G)

    adjacency_matrix = np.zeros([num_nodes, num_nodes])
    for i in range(num_nodes):
        for j in range(num_nodes):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                adjacency_matrix[i, j] = temp['weight']

    return Maxcut(adjacency_matrix).to_quadratic_program()

def get_qaoa_uncompiled(num_qubits:int):
    qc_prep = QuantumCircuit(num_qubits)
    qc_prep.h(range(num_qubits))
    qc_prep.barrier()
    p = Parameter('a')
    qc_prep.rzz(p, 0,1)
    qc_prep.rzz(p, 0,2)
    qc_prep.barrier()

    p = Parameter('b')
    qc_mix = QuantumCircuit(num_qubits)
    qc_mix.rx(2*p, range(num_qubits))
    return qc_prep, qc_mix

def get_mapping(qc:QuantumCircuit):
    coupling_map = FakeManila().configuration().coupling_map
    offline_mapped_qc = transpile(qc, coupling_map=coupling_map, basis_gates=["cx", "sx", "x", "rz"],
                                  optimization_level=3)

    layout = offline_mapped_qc._layout.initial_layout
    mapping = []
    for elem in layout.get_virtual_bits():
        if elem.register.name == "ancilla":
            pass
        else:
            mapping.append(layout.get_virtual_bits()[elem])

    return mapping

def get_qaoa_partially_compiled(num_qubits:int):
    qc_prep_uncompiled, qc_mix_uncompiled = get_qaoa_uncompiled(num_qubits)
    mapping = get_mapping(qc_prep_uncompiled.compose(qc_mix_uncompiled))

    qc_prep_compiled = transpile(qc_prep_uncompiled, basis_gates=["cx", "sx", "x", "rz"], initial_layout=mapping, layout_method="trivial",
                   optimization_level=3)

    qc_mixer_compiled = transpile(qc_mix_uncompiled, basis_gates=["cx", "sx", "x", "rz"], initial_layout=mapping, layout_method="trivial",
                         optimization_level=3)

    return qc_prep_compiled, qc_mixer_compiled
