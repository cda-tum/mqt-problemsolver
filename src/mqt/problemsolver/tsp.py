import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.circuit.library import QFT

from mqt import ddsim


class TSP:
    def print(self, solution=None):
        self.G = nx.DiGraph(directed=True)
        self.G.add_node(1)
        self.G.add_node(2)
        self.G.add_node(3)
        self.G.add_node(4)

        self.G.add_edge(1, 2)
        self.G.add_edge(1, 3)
        self.G.add_edge(1, 4)

        self.G.add_edge(2, 1)
        self.G.add_edge(2, 3)
        self.G.add_edge(2, 4)

        self.G.add_edge(3, 1)
        self.G.add_edge(3, 2)
        self.G.add_edge(3, 4)

        self.G.add_edge(4, 1)
        self.G.add_edge(4, 2)
        self.G.add_edge(4, 3)

        if hasattr(self, "dist_1_2"):
            dist_1_2 = self.dist_1_2
        else:
            dist_1_2 = "dist_1_2"

        if hasattr(self, "dist_1_3"):
            dist_1_3 = self.dist_1_3
        else:
            dist_1_3 = "dist_1_3"

        if hasattr(self, "dist_1_4"):
            dist_1_4 = self.dist_1_4
        else:
            dist_1_4 = "dist_1_4"

        if hasattr(self, "dist_2_3"):
            dist_2_3 = self.dist_2_3
        else:
            dist_2_3 = "dist_2_3"

        if hasattr(self, "dist_2_4"):
            dist_2_4 = self.dist_2_4
        else:
            dist_2_4 = "dist_2_4"

        if hasattr(self, "dist_3_4"):
            dist_3_4 = self.dist_3_4
        else:
            dist_3_4 = "dist_3_4"

        edge_labels = {
            (1, 2): dist_1_2,
            (1, 3): dist_1_3,
            (1, 4): dist_1_4,
            (2, 3): dist_2_3,
            (2, 4): dist_2_4,
            (3, 4): dist_3_4,
        }

        self.pos = {1: [0, 1], 2: [0, 0], 3: [1, 1], 4: [1, 0]}

        nx.draw(
            self.G,
            with_labels=True,
            node_color="skyblue",
            edge_cmap=plt.cm.Blues,
            pos=self.pos,
            node_size=2000,
            font_size=20,
        )

        if solution is not None:
            selected_graph_quantum = self.extract_selected_graph(solution)

            edges_quantum = selected_graph_quantum.edges()
            colors_quantum = [
                selected_graph_quantum[u][v]["color"] for u, v in edges_quantum
            ]
            weights_quantum = [
                selected_graph_quantum[u][v]["weight"] for u, v in edges_quantum
            ]
            nx.draw(
                selected_graph_quantum,
                self.pos,
                node_color="skyblue",
                edge_color=colors_quantum,
                width=weights_quantum,
                node_size=2000,
                font_size=20,
            )

        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, label_pos=0.3, font_size=20
        )

        return

    def solve(
        self,
        dist_1_2,
        dist_1_3,
        dist_1_4,
        dist_2_3,
        dist_2_4,
        dist_3_4,
        quantum_algorithm="QPE",
        num_qubits_qft=8,
    ):
        if quantum_algorithm == "QPE":
            self.dist_1_2 = dist_1_2
            self.dist_1_3 = dist_1_3
            self.dist_1_4 = dist_1_4
            self.dist_2_3 = dist_2_3
            self.dist_2_4 = dist_2_4
            self.dist_3_4 = dist_3_4

            self.distances_sum = sum(
                [dist_1_2, dist_1_3, dist_1_4, dist_2_3, dist_2_4, dist_3_4]
            )

            self.num_qubits_qft = num_qubits_qft
            sol_perm = self.solve_using_QPE()
            self.print(solution=sol_perm)
            return sol_perm

        else:
            print("ERROR: Selected quantum algorithm is not implemented.")
            return False

    def solve_using_QPE(self):
        phases = self.get_all_phases()

        eigen_values = ["11000110", "10001101", "10000111"]
        all_perms = []
        all_costs = []
        for index_eigenstate in range(len(eigen_values)):

            unit = QuantumRegister(self.num_qubits_qft, "unit")
            eigen = QuantumRegister(8, "eigen")
            unit_classical = ClassicalRegister(self.num_qubits_qft, "unit_classical")

            qc = self.create_TSP_qc(
                unit, eigen, unit_classical, index_eigenstate, eigen_values, phases
            )

            most_frequent = self.simulate(qc)

            route = self.eigenvalue_to_route(eigen_values[index_eigenstate])

            most_frequent_decimal = int(most_frequent, 2)
            phase = most_frequent_decimal / (2**self.num_qubits_qft)
            costs = self.phase_to_int(phase)

            all_perms.append(route)
            all_costs.append(costs)

        sol_perm = all_perms[np.argmin(all_costs)]
        return sol_perm

    def create_TSP_qc(
        self, unit, eigen, unit_classical, index_eigenstate, eigen_values, phases
    ):
        qc = QuantumCircuit(unit, eigen, unit_classical)
        self.eigenstates(qc, eigen, index_eigenstate, eigen_values)

        qc.h(unit[:])
        qc.barrier()

        for i in range(0, self.num_qubits_qft):
            qc.append(
                self.final_U(i, eigen, phases),
                [unit[self.num_qubits_qft - 1 - i]] + eigen[:],
            )

        # Inverse QFT
        qc.barrier()
        qft = QFT(
            num_qubits=len(unit),
            inverse=True,
            insert_barriers=True,
            do_swaps=False,
            name="Inverse QFT",
        )
        qc.append(qft, qc.qubits[: len(unit)])
        qc.barrier()

        # Measure
        qc.measure(unit, unit_classical)
        return qc

    def get_all_phases(self):
        a = self.int_to_phase(self.dist_1_2)
        d = a
        b = self.int_to_phase(self.dist_1_3)
        g = b
        c = self.int_to_phase(self.dist_1_4)
        j = c
        e = self.int_to_phase(self.dist_2_3)
        h = e
        f = self.int_to_phase(self.dist_2_4)
        k = f
        i = self.int_to_phase(self.dist_3_4)
        m = i
        return [a, b, c, d, e, f, g, h, i, j, k, m]

    def simulate(self, qc: QuantumCircuit):

        backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        job = execute(qc, backend, shots=1000)
        count = job.result().get_counts()

        return count.most_frequent()

    def get_classical_result(self):
        distance_matrix = np.array(
            [
                [0, self.dist_1_2, self.dist_1_3, self.dist_1_4],
                [self.dist_1_2, 0, self.dist_2_3, self.dist_2_4],
                [self.dist_1_3, self.dist_2_3, 0, self.dist_3_4],
                [self.dist_1_4, self.dist_1_3, self.dist_3_4, 0],
            ]
        )
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

        return (np.array(permutation) + 1).T

    def controlled_unitary(
        self, qc, qubits: list, phases: list
    ):  # x,y,z = Specific Qubit; a,b,c,d = Phases
        qc.cp(phases[2] - phases[0], qubits[0], qubits[1])  # controlled-U1(c-a)
        qc.p(phases[0], qubits[0])  # U1(a)
        qc.cp(phases[1] - phases[0], qubits[0], qubits[2])  # controlled-U1(b-a)

        # controlled controlled U1(d-c+a-b)
        qc.cp((phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[1], qubits[2])
        qc.cx(qubits[0], qubits[1])
        qc.cp(
            -(phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[1], qubits[2]
        )
        qc.cx(qubits[0], qubits[1])
        qc.cp((phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[0], qubits[2])

    def U(
        self, times, qc, unit, eigen, phases: list
    ):  # a,b,c = phases for U1; d,e,f = phases for U2; g,h,i = phases for U3; j,k,l = phases for U4; m_list=[m, n, o, p, q, r, s, t, u, a, b, c, d, e, f, g, h, i, j, k, l]
        self.controlled_unitary(qc, [unit[0]] + eigen[0:2], [0] + phases[0:3])
        self.controlled_unitary(
            qc, [unit[0]] + eigen[2:4], [phases[3]] + [0] + phases[4:6]
        )
        self.controlled_unitary(
            qc, [unit[0]] + eigen[4:6], phases[6:8] + [0] + [phases[8]]
        )
        self.controlled_unitary(qc, [unit[0]] + eigen[6:8], phases[9:12] + [0])

    def final_U(self, times, eigen, phases: list):
        unit = QuantumRegister(1, "unit")
        qc = QuantumCircuit(unit, eigen)
        for _ in range(2**times):
            self.U(times, qc, unit, eigen, phases)
        return qc.to_gate(label="U" + "_" + (str(2**times)))

    # Function to place appropriate corresponding gate according to eigenstates
    def eigenstates(self, qc, eigen, index, eigen_values):
        for i in range(0, len(eigen)):
            if eigen_values[index][i] == "1":
                qc.x(eigen[i])
            if eigen_values[index][i] == "0":
                pass
        qc.barrier()
        return qc

    def int_to_phase(self, distance):
        phase = distance / self.distances_sum * 2 * np.pi
        return phase

    def phase_to_int(self, phase):
        return phase * self.distances_sum

    def eigenvalue_to_route(self, eigenvalue: str):
        a = int(eigenvalue[0:2], 2) + 1
        b = int(eigenvalue[2:4], 2) + 1
        c = int(eigenvalue[4:6], 2) + 1
        d = int(eigenvalue[6:8], 2) + 1
        return [a, b, c, d]

    def extract_selected_graph(self, solution):
        G = nx.Graph()
        for i in range(len(solution)):
            if i == len(solution) - 1:
                G.add_edge(solution[i], solution[0], color="r", weight=2)
            else:
                G.add_edge(solution[i], solution[i + 1], color="r", weight=2)

        return G

    def get_available_quantum_algorithms(self):
        return ["QPE"]

    def show_classical_solution(self):
        self.print(solution=self.get_classical_result())
