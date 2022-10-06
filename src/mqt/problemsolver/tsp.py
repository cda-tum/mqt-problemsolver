# Code is adapted from https://arxiv.org/abs/1805.10928 and https://qiskit.org/textbook/ch-paper-implementations/tsp.html
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.circuit.library import QFT

from mqt import ddsim


class TSP:
    def print(self, solution: list[int] = None):
        """Method to visualize the problem.

        Keyword arguments:
        solution -- If provided, the solution is visualized. Otherwise, the problem without solution is shown.

        """
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

        pos = {1: [0, 1], 2: [0, 0], 3: [1, 1], 4: [1, 0]}

        nx.draw(
            G,
            with_labels=True,
            node_color="skyblue",
            edge_cmap=plt.cm.Blues,
            pos=pos,
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
                pos,
                node_color="skyblue",
                edge_color=colors_quantum,
                width=weights_quantum,
                node_size=2000,
                font_size=20,
            )

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=20
        )

        return

    def solve(
        self,
        dist_1_2: int,
        dist_1_3: int,
        dist_1_4: int,
        dist_2_3: int,
        dist_2_4: int,
        dist_3_4: int,
        objective_function="shortest_path",
        quantum_algorithm="QPE",
        num_qubits_qft=8,
    ):
        """Method to solve the problem.

        Keyword arguments:
        dist_*_* -- Defining the adjacency matrix by the distances between the vertices.
        objective_function -- Optimization goal.
        quantum_algorithm -- Selected quantum algorithm to solve problem.
        num_qubits_qft -- Number of qubits used for QFT if "QPE" is selected as the algorithm.

        Return values:
        solution -- Solution to the problem if it exists.
        """
        if quantum_algorithm == "QPE" and objective_function == "shortest_path":
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
            return sol_perm

        else:
            print(
                "ERROR: Combination of objective function quantum algorithm is not implemented."
            )
            return False

    def solve_using_QPE(self):

        eigen_values = ["11000110", "10001101", "10000111"]
        all_perms = []
        all_costs = []
        for eigenstate in eigen_values:

            qft_register = QuantumRegister(self.num_qubits_qft, "qft")
            eigenstate_register = QuantumRegister(8, "eigen")
            qft_register_classical = ClassicalRegister(
                self.num_qubits_qft, "qft_classical"
            )

            qc = self.create_TSP_qc(
                qft_register, eigenstate_register, qft_register_classical, eigenstate
            )

            most_frequent = self.simulate(qc)

            route = self.eigenvalue_to_route(eigenstate)

            most_frequent_decimal = int(most_frequent, 2)
            phase = most_frequent_decimal / (2**self.num_qubits_qft)
            costs = self.phase_to_int(phase)

            all_perms.append(route)
            all_costs.append(costs)

        solution = all_perms[np.argmin(all_costs)]
        return solution

    def create_TSP_qc(
        self,
        qft_register: QuantumRegister,
        eigenstate_register: QuantumRegister,
        qft_classical_register: ClassicalRegister,
        eigenstate: QuantumRegister,
    ):
        qc = QuantumCircuit(qft_register, eigenstate_register, qft_classical_register)

        self.encode_eigenstate(qc, eigenstate_register, eigenstate)

        qc.h(qft_register[:])
        qc.barrier()

        for i in range(0, self.num_qubits_qft):
            qc.append(
                self.final_U(times=i, eigenstate_register=eigenstate_register),
                [qft_register[self.num_qubits_qft - 1 - i]] + eigenstate_register[:],
            )

        # Inverse QFT
        qc.barrier()
        qft = QFT(
            num_qubits=len(qft_register),
            inverse=True,
            insert_barriers=True,
            do_swaps=False,
            name="Inverse QFT",
        )
        qc.append(qft, qc.qubits[: len(qft_register)])
        qc.barrier()

        # Measure
        qc.measure(qft_register, qft_classical_register)
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
        self, qc: QuantumCircuit, qubits: list[QuantumRegister], phases: list[int]
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

        # Alternative formulation
        # from qiskit.extensions import UnitaryGate
        #
        # matrix = [
        #     [1, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 1, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0 + np.exp(0 + phases[0] * 1j), 0, 0, 0],
        #     [0, 0, 0, 0, 0, np.exp(phases[1] * 1j), 0, 0],
        #     [0, 0, 0, 0, 0, 0, np.exp(phases[2] * 1j), 0],
        #     [0, 0, 0, 0, 0, 0, 0, np.exp(phases[3] * 1j)],
        # ]
        #
        # gate = UnitaryGate(matrix)
        # qc.append(gate, qubits)

    def U(
        self,
        qc: QuantumCircuit,
        control_qreg: QuantumRegister,
        eigenstate_register: QuantumRegister,
    ):

        phases = self.get_all_phases()
        # a,b,c = phases for U1; d,e,f = phases for U2; g,h,i = phases for U3; j,k,l = phases for U4;
        self.controlled_unitary(
            qc, [control_qreg[0]] + eigenstate_register[0:2], [0] + phases[0:3]
        )
        self.controlled_unitary(
            qc,
            [control_qreg[0]] + eigenstate_register[2:4],
            [phases[3]] + [0] + phases[4:6],
        )
        self.controlled_unitary(
            qc,
            [control_qreg[0]] + eigenstate_register[4:6],
            phases[6:8] + [0] + [phases[8]],
        )
        self.controlled_unitary(
            qc, [control_qreg[0]] + eigenstate_register[6:8], phases[9:12] + [0]
        )

    def final_U(self, times: int, eigenstate_register: QuantumRegister):
        control_qreg = QuantumRegister(1, "control")
        qc = QuantumCircuit(control_qreg, eigenstate_register)
        for _ in range(2**times):
            self.U(qc, control_qreg, eigenstate_register)
        return qc.to_gate(label="U" + "_" + (str(2**times)))

    def encode_eigenstate(
        self,
        qc: QuantumCircuit,
        eigen_register: QuantumRegister,
        eigenstate: QuantumRegister,
    ):
        for i in range(0, len(eigen_register)):
            if eigenstate[i] == "1":
                qc.x(eigen_register[i])
            if eigenstate[i] == "0":
                pass
        qc.barrier()
        return qc

    def int_to_phase(self, distance: int):
        phase = distance / self.distances_sum * 2 * np.pi
        return phase

    def phase_to_int(self, phase: float):
        return phase * self.distances_sum

    def eigenvalue_to_route(self, eigenvalue: str):
        a = int(eigenvalue[0:2], 2) + 1
        b = int(eigenvalue[2:4], 2) + 1
        c = int(eigenvalue[4:6], 2) + 1
        d = int(eigenvalue[6:8], 2) + 1
        return [a, b, c, d]

    def extract_selected_graph(self, solution: list[int]):
        G = nx.Graph()
        for i in range(len(solution)):
            if i == len(solution) - 1:
                G.add_edge(solution[i], solution[0], color="r", weight=2)
            else:
                G.add_edge(solution[i], solution[i + 1], color="r", weight=2)

        return G

    def get_available_quantum_algorithms(self):
        """Method to get all available quantum algorithms in a list."""
        return ["QPE"]

    def show_classical_solution(self):
        """Method to visualize the solution of a classical solver."""
        self.print(solution=self.get_classical_result())
