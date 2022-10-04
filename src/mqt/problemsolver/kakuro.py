import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, execute

from mqt import ddsim


class Kakuro:
    def solve(
        self,
        s0_input: int,
        s1_input: int,
        s2_input: int,
        s3_input: int,
        quantum_algorithm="Grover",
    ):
        self.s0_input = s0_input
        self.s1_input = s1_input
        self.s2_input = s2_input
        self.s3_input = s3_input

        if quantum_algorithm == "Grover":
            qc, anc, anc_mct, flag, nqubits, nancilla, (a, b, c, d) = self.init_qc()
            qc, mct_list = self.encode_constraints(qc, a, b, c, d, anc)
            oracle = self.create_oracle(qc, mct_list, flag, anc_mct)
            for m in (5, 6, 7, 8, 12):
                qc = self.create_grover(
                    oracle, nqubits, nancilla, ninputs=nqubits - 1, grover_iterations=m
                )
                res = self.simulate(qc)
                if res:
                    break

            self.print(a=str(res[0]), b=str(res[1]), c=str(res[2]), d=str(res[3]))
            return res

        else:
            print("ERROR: Selected quantum algorithm is not implemented.")
            return False

    def print(self, a="a", b="b", c="c", d="d"):

        if hasattr(self, "s0_input"):
            s0_input = self.s0_input
        else:
            s0_input = "s0"

        if hasattr(self, "s1_input"):
            s1_input = self.s1_input
        else:
            s1_input = "s1"

        if hasattr(self, "s2_input"):
            s2_input = self.s2_input
        else:
            s2_input = "s2"

        if hasattr(self, "s3_input"):
            s3_input = self.s3_input
        else:
            s3_input = "s3"

        print("     | ", s0_input, " | ", s1_input, " |")
        print("------------------")
        print(" ", s2_input, " | ", a, " | ", b, " |")
        print("------------------")
        print(" ", s3_input, " | ", c, " | ", d, " |")
        print("------------------\n")

    def check_inequality(self, qc, x, y, res_anc):
        x_low, x_high = x
        y_low, y_high = y

        qc.cx(x_high, y_high)
        qc.x(y_high)
        qc.cx(x_low, y_low)
        qc.x(y_low)

        qc.rccx(y_low, y_high, res_anc)
        qc.x(res_anc)

        # Uncompute
        qc.x(y_low)
        qc.cx(x_low, y_low)
        qc.x(y_high)
        qc.cx(x_high, y_high)

    def check_equality(self, qc, x, s, res_anc):
        x_low, x_mid, x_high = x

        if s[-1] == "0":
            qc.x(x_low)
        if s[-2] == "0":
            qc.x(x_mid)
        if s[-3] == "0":
            qc.x(x_high)

        qc.rcccx(x_low, x_mid, x_high, res_anc)

    def add_two_numbers(self, qc, x, y, ancs, res_anc_low, res_anc_high, anc_carry):
        x_low, x_high = x
        y_low, y_high = y

        qc.rccx(x_low, y_low, res_anc_high)
        qc.cx(x_low, y_low)
        qc.cx(y_low, res_anc_low)
        qc.rccx(x_high, y_high, ancs[0])
        qc.cx(x_high, y_high)
        qc.rccx(y_high, res_anc_high, ancs[1])
        qc.cx(y_high, res_anc_high)

        qc.x(ancs[0])
        qc.x(ancs[1])
        qc.x(anc_carry)
        qc.rccx(ancs[0], ancs[1], anc_carry)

        # Uncompute
        qc.cx(x_high, y_high)
        qc.cx(x_low, y_low)

        return (res_anc_low, res_anc_high, anc_carry)

    def encode_constraints(self, qc: QuantumCircuit, a, b, c, d, anc):
        mct_list = []

        # Inequalities
        self.check_inequality(qc, a, b, anc[0])
        mct_list.append(anc[0])
        qc.barrier()
        self.check_inequality(qc, b, d, anc[1])
        mct_list.append(anc[1])
        qc.barrier()
        self.check_inequality(qc, d, c, anc[2])
        mct_list.append(anc[2])
        qc.barrier()
        self.check_inequality(qc, c, a, anc[3])
        mct_list.append(anc[3])
        qc.barrier()

        # Equalities
        tmp_1 = self.add_two_numbers(qc, a, b, anc[4:6], anc[6], anc[7], anc[8])
        qc.barrier()
        self.check_equality(qc, tmp_1, bin(self.s2_input)[2:].zfill(3), anc[9])
        mct_list.append(anc[9])
        qc.barrier()

        tmp_2 = self.add_two_numbers(qc, c, d, anc[10:12], anc[12], anc[13], anc[14])
        qc.barrier()
        self.check_equality(qc, tmp_2, bin(self.s3_input)[2:].zfill(3), anc[15])
        mct_list.append(anc[15])
        qc.barrier()

        tmp_3 = self.add_two_numbers(qc, b, d, anc[16:18], anc[18], anc[19], anc[20])
        qc.barrier()
        self.check_equality(qc, tmp_3, bin(self.s1_input)[2:].zfill(3), anc[21])
        mct_list.append(anc[21])
        qc.barrier()

        tmp_4 = self.add_two_numbers(qc, a, c, anc[22:24], anc[24], anc[25], anc[26])
        qc.barrier()
        self.check_equality(qc, tmp_4, bin(self.s0_input)[2:].zfill(3), anc[27])
        mct_list.append(anc[27])

        return (qc, mct_list)

    def create_oracle(self, qc: QuantumCircuit, mct_list, flag, anc_mct):
        compute = qc.to_instruction()

        # mark solution
        qc.mct(mct_list, flag, ancilla_qubits=anc_mct, mode="v-chain")

        # uncompute
        uncompute = compute.inverse()
        uncompute.name = "uncompute"
        qc.append(uncompute, range(qc.num_qubits))

        oracle = qc.to_instruction(label="oracle")
        return oracle

    def init_qc(self):
        a_low = QuantumRegister(1, "a_low")
        a_high = QuantumRegister(1, "a_high")
        a = (a_low, a_high)
        b_low = QuantumRegister(1, "b_low")
        b_high = QuantumRegister(1, "b_high")
        b = (b_low, b_high)
        c_low = QuantumRegister(1, "c_low")
        c_high = QuantumRegister(1, "c_high")
        c = (c_low, c_high)
        d_low = QuantumRegister(1, "d_low")
        d_high = QuantumRegister(1, "d_high")
        d = (d_low, d_high)

        anc = QuantumRegister(28, "anc")
        anc_mct = QuantumRegister(10, "mct_ancilla")
        flag = QuantumRegister(1, "flag")
        qc = QuantumCircuit(
            a_low,
            a_high,
            b_low,
            b_high,
            c_low,
            c_high,
            d_low,
            d_high,
            anc,
            anc_mct,
            flag,
        )

        nqubits = 9
        nancilla = anc.size + anc_mct.size
        return (qc, anc, anc_mct, flag, nqubits, nancilla, (a, b, c, d))

    def create_grover(self, oracle, nqubits, nancilla, ninputs, grover_iterations):
        import numpy as np

        qc = QuantumCircuit(nqubits + nancilla, ninputs)
        qc.h(range(ninputs))
        qc.x(nqubits + nancilla - 1)
        qc.h(nqubits + nancilla - 1)

        for _ in range(round(grover_iterations)):
            qc.append(oracle, range(nqubits + nancilla))
            qc.h(range(ninputs))
            qc.x(range(ninputs))
            qc.mcp(np.pi, list(range(ninputs - 1)), ninputs - 1)
            qc.x(range(ninputs))
            qc.h(range(ninputs))
        qc.measure(range(ninputs), range(ninputs))

        return qc

    def simulate(self, qc):
        backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")
        job = execute(qc, backend, shots=10000)
        counts = job.result().get_counts(qc)

        mean_counts = np.mean(list(counts.values()))

        found_sol = False
        for entry in counts.keys():

            if counts.get(entry) > 5 * mean_counts:
                found_sol = True
                break
        if found_sol:
            for entry in counts.keys():
                d = int(entry[0:2], 2)
                c = int(entry[2:4], 2)
                b = int(entry[4:6], 2)
                a = int(entry[6:8], 2)
                if counts.get(entry) > 5 * mean_counts:
                    return (a, b, c, d)
        else:
            print("Sums are impossible to satisfy. Please try another setup.")

    def get_available_quantum_algorithms(self):
        return ["Grover"]
