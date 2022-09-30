from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram, array_to_latex
from qiskit.circuit.library import QFT
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
import networkx as nx
import matplotlib.pyplot as plt
from mqt import ddsim


class TSP:
    def solve(
        self, dist_1_2, dist_1_3, dist_1_4, dist_2_3, dist_2_4, dist_3_4, num_qubits_qft
    ):
        self.dist_1_2 = dist_1_2
        self.dist_1_3 = dist_1_3
        self.dist_1_4 = dist_1_4
        self.dist_2_3 = dist_2_3
        self.dist_2_4 = dist_2_4
        self.dist_3_4 = dist_3_4

        self.distances_sum = sum(
            [dist_1_2, dist_1_3, dist_1_4, dist_2_3, dist_2_4, dist_3_4]
        )
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

        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, label_pos=0.3, font_size=20
        )

        plt.show()

        sol_perm = self.create_qc(num_qubits_qft)
        self.visualize_solution(sol_perm)

    def create_qc(self, num_qubits_qft):
        # Assign phase to each edge
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
        l = i

        # Storing the eigenvalues in a list
        eigen_values = ["11000110", "10001101", "10000111"]
        all_perms = []
        all_costs = []
        for index_eigenstate in (0, 1, 2):
            # Initialization
            unit = QuantumRegister(num_qubits_qft, "unit")
            eigen = QuantumRegister(8, "eigen")
            unit_classical = ClassicalRegister(num_qubits_qft, "unit_classical")
            qc = QuantumCircuit(unit, eigen, unit_classical)

            self.eigenstates(qc, eigen, index_eigenstate, eigen_values)

            qc.h(unit[:])
            qc.barrier()

            # Controlled Unitary
            phases = [a, b, c, d, e, f, g, h, i, j, k, l]
            for i in range(0, num_qubits_qft):
                qc.append(
                    self.final_U(i, eigen, phases),
                    [unit[num_qubits_qft - 1 - i]] + eigen[:],
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

            most_frequent = self.simulate(qc)

            most_frequent_decimal = int(most_frequent, 2)
            phase = most_frequent_decimal / (2**num_qubits_qft)
            route = self.eigenvalue_to_route(eigen_values[index_eigenstate])
            costs = self.phase_to_int(phase)
            print("Eigenstate: ", eigen_values[index_eigenstate])
            print("Most frequent Binary: ", most_frequent)
            print("Most frequent Decimal: ", most_frequent_decimal)
            print("Phase: ", phase)
            print("Route: ", route)
            print("Costs: ", costs)

            all_perms.append(route)
            all_costs.append(costs)

        sol_perm = all_perms[np.argmin(all_costs)]
        print("### Solution: ", sol_perm)
        return sol_perm

    def simulate(self, qc: QuantumCircuit):

        backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        job = execute(qc, backend, shots=1000)
        count = job.result().get_counts()

        return count.most_frequent()

    def visualize_solution(self, sol_perm):
        edge_labels = {
            (1, 2): self.dist_1_2,
            (1, 3): self.dist_1_3,
            (1, 4): self.dist_1_4,
            (2, 3): self.dist_2_3,
            (2, 4): self.dist_2_4,
            (3, 4): self.dist_3_4,
        }
        selected_graph_quantum = self.extract_selected_graph(sol_perm)

        edges_quantum = selected_graph_quantum.edges()
        colors_quantum = [
            selected_graph_quantum[u][v]["color"] for u, v in edges_quantum
        ]
        weights_quantum = [
            selected_graph_quantum[u][v]["weight"] for u, v in edges_quantum
        ]

        plt.title("Quantum Solution")
        nx.draw(
            self.G,
            with_labels=True,
            node_color="skyblue",
            edge_cmap=plt.cm.Blues,
            pos=self.pos,
            node_size=2000,
            font_size=20,
        )
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

    def print_problem():
        G = nx.DiGraph(directed=True)
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_node(4)

        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 4)

        G.add_edge(2, 1)
        G.add_edge(2, 3)
        G.add_edge(2, 4)

        G.add_edge(3, 1)
        G.add_edge(3, 2)
        G.add_edge(3, 4)

        G.add_edge(4, 1)
        G.add_edge(4, 2)
        G.add_edge(4, 3)

        pos = {1: [0, 1], 2: [0, 0], 3: [1, 1], 4: [1, 0]}

        edge_labels = {
            (1, 2): "1<->2",
            (1, 3): "1<->3",
            (1, 4): "1<->4",
            (2, 3): "2<->3",
            (2, 4): "2<->4",
            (3, 4): "3<->4",
        }

        nx.draw(
            G,
            with_labels=True,
            node_color="skyblue",
            pos=pos,
            node_size=2000,
            font_size=20,
        )

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=20
        )

        plt.show()
        return

    def show_classical_result(self):
        distance_matrix = np.array(
            [
                [0, self.dist_1_2, self.dist_1_3, self.dist_1_4],
                [self.dist_1_2, 0, self.dist_2_3, self.dist_2_4],
                [self.dist_1_3, self.dist_2_3, 0, self.dist_3_4],
                [self.dist_1_4, self.dist_1_3, self.dist_3_4, 0],
            ]
        )
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
        edge_labels = {
            (1, 2): self.dist_1_2,
            (1, 3): self.dist_1_3,
            (1, 4): self.dist_1_4,
            (2, 3): self.dist_2_3,
            (2, 4): self.dist_2_4,
            (3, 4): self.dist_3_4,
        }

        selected_graph_classical = self.extract_selected_graph(
            np.array(permutation) + 1
        )
        edges_classical = selected_graph_classical.edges()
        colors_classical = [
            selected_graph_classical[u][v]["color"] for u, v in edges_classical
        ]
        weights_classical = [
            selected_graph_classical[u][v]["weight"] for u, v in edges_classical
        ]

        plt.title("Classical Solution")
        nx.draw(
            self.G,
            with_labels=True,
            node_color="skyblue",
            edge_cmap=plt.cm.Blues,
            pos=self.pos,
            node_size=2000,
            font_size=20,
        )
        nx.draw(
            selected_graph_classical,
            self.pos,
            node_color="skyblue",
            edge_color=colors_classical,
            width=weights_classical,
            node_size=2000,
            font_size=20,
        )
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, label_pos=0.3, font_size=20
        )
        plt.show()

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
