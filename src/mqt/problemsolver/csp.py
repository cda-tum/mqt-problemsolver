from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from mqt.core import load
from mqt.core.dd import sample
from qiskit import QuantumCircuit, QuantumRegister

if TYPE_CHECKING:
    from qiskit.circuit import Instruction


class Constraint(TypedDict, total=False):
    """Class to store the properties of a single constraint."""

    constraint_type: str
    operand_one: str
    operand_two: str
    to_be_satisfied_sum: int


class CSP:
    def solve(
        self,
        constraints: list[Constraint],
        quantum_algorithm: str = "Grover",
    ) -> tuple[int, int, int, int] | bool:
        """Method to solve the problem.

        Args:
            constraints: List of to be satisfied constraints.
            quantum_algorithm: Selected quantum algorithm to solve problem.

        Returns:
            Solution to the problem if it exists.
        """
        if quantum_algorithm == "Grover":
            qc, anc, flag, nqubits, nancilla, (a, b, c, d) = self.init_qc()
            qc, mct_list = self.encode_constraints(qc, a, b, c, d, anc, constraints=constraints)
            oracle = self.create_oracle(qc, mct_list, flag)
            for m in (5, 6, 7, 8, 12):
                qc = self.create_grover(oracle, nqubits, nancilla, ninputs=nqubits - 1, grover_iterations=m)
                solution = self.simulate(qc)
                if solution:
                    break
            if solution:
                return solution
            return False

        print("ERROR: Selected quantum algorithm is not implemented.")
        return False

    def print_problem(
        self,
        sum_s0: str | int = "s0",
        sum_s1: str | int = "s1",
        sum_s2: str | int = "s2",
        sum_s3: str | int = "s3",
        a: str | int = "a",
        b: str | int = "b",
        c: str | int = "c",
        d: str | int = "d",
    ) -> None:
        """Method to visualize the problem.

        Args:
            sum_s0: Sum to be satisfied.
            sum_s1: Sum to be satisfied.
            sum_s2: Sum to be satisfied.
            sum_s3: Sum to be satisfied.
            a: Variable value satisfying the respective sums.
            b: Variable value satisfying the respective sums.
            c: Variable value satisfying the respective sums.
            d: Variable value satisfying the respective sums.
        """
        print("     | ", sum_s0, " | ", sum_s1, " | ")
        print("------------------")
        print(" ", sum_s2, " | ", a, " | ", b, " |")
        print("------------------")
        print(" ", sum_s3, " | ", c, " | ", d, " |")
        print("------------------\n")

    def check_inequality(
        self,
        qc: QuantumCircuit,
        x: tuple[QuantumRegister, QuantumRegister],
        y: tuple[QuantumRegister, QuantumRegister],
        res_anc: QuantumRegister,
    ) -> None:
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

    def check_equality(
        self,
        qc: QuantumCircuit,
        x: tuple[QuantumRegister, QuantumRegister, QuantumRegister],
        s: str,
        res_anc: QuantumRegister,
    ) -> None:
        x_low, x_mid, x_high = x

        if s[-1] == "0":
            qc.x(x_low)
        if s[-2] == "0":
            qc.x(x_mid)
        if s[-3] == "0":
            qc.x(x_high)

        qc.rcccx(x_low, x_mid, x_high, res_anc)

    def add_two_numbers(
        self,
        qc: QuantumCircuit,
        x: tuple[QuantumRegister, QuantumRegister],
        y: tuple[QuantumRegister, QuantumRegister],
        ancs: QuantumRegister,
        res_anc_low: QuantumRegister,
        res_anc_high: QuantumRegister,
        anc_carry: QuantumRegister,
    ) -> tuple[QuantumRegister, QuantumRegister, QuantumRegister]:
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

    def encode_constraints(
        self,
        qc: QuantumCircuit,
        a: QuantumRegister,
        b: QuantumRegister,
        c: QuantumRegister,
        d: QuantumRegister,
        anc: QuantumRegister,
        constraints: list[Constraint],
    ) -> tuple[QuantumCircuit, list[QuantumRegister]]:
        mct_list = []

        dict_variable_to_quantumregister = {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
        }
        anc_needed_per_constraint = {
            "inequality": 1,
            "addition_equality": 6,
        }
        anc_index = 0
        for constraint in constraints:
            if constraint.get("constraint_type") == "inequality":
                first_qreg = dict_variable_to_quantumregister[constraint["operand_one"]]
                second_qreg = dict_variable_to_quantumregister[constraint["operand_two"]]

                self.check_inequality(qc, first_qreg, second_qreg, anc[anc_index])
                mct_list.append(anc[anc_index])
                qc.barrier()
                anc_index += anc_needed_per_constraint[constraint["constraint_type"]]

            elif constraint.get("constraint_type") == "addition_equality":
                first_qreg = dict_variable_to_quantumregister[constraint["operand_one"]]
                second_qreg = dict_variable_to_quantumregister[constraint["operand_two"]]

                tmp_1 = self.add_two_numbers(
                    qc,
                    first_qreg,
                    second_qreg,
                    anc[anc_index : anc_index + 2],
                    anc[anc_index + 2],
                    anc[anc_index + 3],
                    anc[anc_index + 4],
                )
                qc.barrier()
                self.check_equality(
                    qc,
                    tmp_1,
                    bin(constraint["to_be_satisfied_sum"])[2:].zfill(3),
                    anc[anc_index + 5],
                )
                mct_list.append(anc[anc_index + 5])
                qc.barrier()
                anc_index += anc_needed_per_constraint[constraint["constraint_type"]]
            else:
                print("Unexpected constraint type: ", constraint["constraint_type"])

        return (qc, mct_list)

    def create_oracle(
        self,
        qc: QuantumCircuit,
        mct_list: list[QuantumRegister],
        flag: QuantumRegister,
    ) -> Instruction:
        compute = qc.to_instruction()

        # mark solution
        qc.mcx(mct_list, flag)

        # uncompute
        uncompute = compute.inverse()
        uncompute.name = "uncompute"
        qc.append(uncompute, range(qc.num_qubits))

        return qc.to_instruction(label="oracle")

    def init_qc(
        self,
    ) -> tuple[
        QuantumCircuit,
        QuantumRegister,
        QuantumRegister,
        int,
        int,
        tuple[
            tuple[QuantumRegister, QuantumRegister],
            tuple[QuantumRegister, QuantumRegister],
            tuple[QuantumRegister, QuantumRegister],
            tuple[QuantumRegister, QuantumRegister],
        ],
    ]:
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
            flag,
        )

        nqubits = 9
        nancilla = anc.size
        return (qc, anc, flag, nqubits, nancilla, (a, b, c, d))

    def create_grover(
        self,
        oracle: QuantumCircuit,
        nqubits: int,
        nancilla: int,
        ninputs: int,
        grover_iterations: int,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(nqubits + nancilla, ninputs)
        qc.h(range(ninputs))
        qc.x(nqubits + nancilla - 1)
        qc.h(nqubits + nancilla - 1)

        for _ in range(grover_iterations):
            qc.append(oracle, range(nqubits + nancilla))
            qc.h(range(ninputs))
            qc.x(range(ninputs))
            qc.mcp(np.pi, list(range(ninputs - 1)), ninputs - 1)
            qc.x(range(ninputs))
            qc.h(range(ninputs))
        qc.measure(range(ninputs), range(ninputs))

        return qc

    def simulate(self, qc: QuantumCircuit) -> tuple[int, int, int, int] | None:
        qc = qc.decompose()
        quantum_computation = load(qc)
        counts = sample(quantum_computation, shots=10000)

        mean_counts = np.mean(list(counts.values()))

        found_sol = False
        for entry in counts:
            if counts.get(entry) > 5 * mean_counts:
                found_sol = True
                break
        if found_sol:
            for entry in counts:
                d = int(entry[0:2], 2)
                c = int(entry[2:4], 2)
                b = int(entry[4:6], 2)
                a = int(entry[6:8], 2)
                if counts.get(entry) > 5 * mean_counts:
                    return (a, b, c, d)

        print("Simulation was unsuccessful.")
        return None

    def get_available_quantum_algorithms(self) -> list[str]:
        """Method to get all available quantum algorithms in a list."""
        return ["Grover"]

    def get_kakuro_constraints(self, sum_s0: int, sum_s1: int, sum_s2: int, sum_s3: int) -> list[Constraint]:
        """Method to get a list of constraints for the inserted sums."""
        list_of_constraints = []
        constraint_1: Constraint = {
            "constraint_type": "addition_equality",
            "operand_one": "a",
            "operand_two": "c",
            "to_be_satisfied_sum": sum_s0,
        }
        list_of_constraints.append(constraint_1)

        constraint_2: Constraint = {
            "constraint_type": "addition_equality",
            "operand_one": "b",
            "operand_two": "d",
            "to_be_satisfied_sum": sum_s1,
        }
        list_of_constraints.append(constraint_2)

        constraint_3: Constraint = {
            "constraint_type": "addition_equality",
            "operand_one": "a",
            "operand_two": "b",
            "to_be_satisfied_sum": sum_s2,
        }
        list_of_constraints.append(constraint_3)

        constraint_4: Constraint = {
            "constraint_type": "addition_equality",
            "operand_one": "c",
            "operand_two": "d",
            "to_be_satisfied_sum": sum_s3,
        }
        list_of_constraints.append(constraint_4)

        constraint_5: Constraint = {
            "constraint_type": "inequality",
            "operand_one": "a",
            "operand_two": "c",
        }
        list_of_constraints.append(constraint_5)

        constraint_6: Constraint = {
            "constraint_type": "inequality",
            "operand_one": "b",
            "operand_two": "d",
        }
        list_of_constraints.append(constraint_6)

        constraint_7: Constraint = {
            "constraint_type": "inequality",
            "operand_one": "a",
            "operand_two": "b",
        }
        list_of_constraints.append(constraint_7)

        constraint_8: Constraint = {
            "constraint_type": "inequality",
            "operand_one": "c",
            "operand_two": "d",
        }
        list_of_constraints.append(constraint_8)
        return list_of_constraints
