from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.providers import BackendV2

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def solve_using_qaoa(qubo: NDArray[np.float64], noisy_flag: bool = False, layers: int = 10, num_init: int = 1) -> float:
    backend = GenericBackendV2(noise_info=noisy_flag, num_qubits=qubo.shape[0])

    qaoa = QAOA(qubo, layers, num_init, backend)
    qc_qaoa, res_qaoa = qaoa.get_solution()
    return res_qaoa


def solve_using_vqe(
    qubo: NDArray[np.float64], noisy_flag: bool = False, ansatz: QuantumCircuit | None = None, num_init: int = 1
) -> float:
    backend = GenericBackendV2(noise_info=noisy_flag, num_qubits=qubo.shape[0])

    vqe = VQE(qubo, backend, ansatz, num_init)
    qc_vqe, res_vqe = vqe.get_solution()
    return res_vqe


def bitstring_to_vector(bitstring: str) -> NDArray[np.int_]:
    """Convert a bitstring to a vector of integers."""
    return np.array([int(bit) for bit in bitstring])


def compute_expectation(counts: dict[str, int], Q: NDArray[np.float64]) -> float:
    """Computes the expectation value of the quadratic form Q with respect to the given counts.

    Args:
        counts: A dictionary where keys are bitstrings and values are their counts.
        Q: The QUBO matrix represented as a NumPy array.

    Returns:
        The expectation value of the quadratic form Q with respect to the counts.
    """
    total_counts = sum(counts.values())
    expectation = 0.0
    for bitstring, count in counts.items():
        x = bitstring_to_vector(bitstring)
        energy = x @ Q @ x  # Compute x^T Q x
        weight = count / total_counts
        expectation += weight * energy

    return expectation


def cost_func(params: list[float], circuit: QuantumCircuit, backend: BackendV2, qubo: NDArray[np.float64]) -> float:
    """Computes the cost function for QAOA optimization.

    Args:
        params: A list of floats representing the parameters for the QAOA circuit
        circuit: The QuantumCircuit representing the QAOA circuit to be evaluated.

    Returns:
        A float representing the expectation value of the QUBO Hamiltonian with respect
        to the states produced by the circuit.
    """
    param_dict = {param: params[i] for i, param in enumerate(circuit.parameters)}
    circuit = circuit.assign_parameters(param_dict)

    job = backend.run(circuit, shots=10000)
    result = job.result()
    counts = result.get_counts()

    return compute_expectation(counts, qubo)


def optimize_with_multiple_init_parameters(
    num_init: int, circuit: QuantumCircuit, backend: BackendV2, qubo: NDArray[np.float64]
) -> tuple[list[float], QuantumCircuit]:
    """Optimizes the QAOA parameters using multiple random initializations.

    Args:
        circuit: The QuantumCircuit representing the QAOA circuit to be optimized.

    Returns:
        A list of floats representing the optimized parameters that minimize the cost function.
    """
    best_cost = float("inf")
    best_params = []

    for _ in range(num_init):
        params = np.random.uniform(0, 2 * np.pi, size=len(circuit.parameters))
        res = minimize(
            cost_func,
            x0=params,
            args=(circuit, backend, qubo),
            method="COBYLA",
            bounds=[(0, 2 * np.pi) for i in range(len(circuit.parameters))],
            options={"disp": False, "maxiter": 1000},
        )

        if res.fun < best_cost:
            best_cost = res.fun
            best_params = res.x
    # Assign the best parameters to the circuit
    param_dict = {param: best_params[i] for i, param in enumerate(circuit.parameters)}
    circuit = circuit.assign_parameters(param_dict)
    return best_params, circuit


def evaluate_result(
    num_init: int, circuit: QuantumCircuit, backend: BackendV2, qubo: NDArray[np.float64]
) -> tuple[str, int, float]:
    """Evaluates the result of the QAOA circuit.

    Args:
        circuit: The QAOA circuit to be evaluated.

    Returns:
        A tuple containing:
        - The found state as a bitstring.
        - The count of that state.
        - The energy computed from the QUBO matrix.
    """
    optimized_params, circuit = optimize_with_multiple_init_parameters(num_init, circuit, backend, qubo)

    job = backend.run(circuit, shots=10000)
    result = job.result()
    counts = result.get_counts()

    found_state = max(counts, key=counts.get)
    x = bitstring_to_vector(found_state)
    energy = x @ qubo @ x

    return found_state, counts[found_state], energy


class QAOA:
    def __init__(self, qubo: NDArray[np.float64], layers: int, num_init: int, backend: BackendV2) -> None:
        self.qubo = qubo
        self.layers = layers
        self.num_init = num_init
        self.backend = backend

    def get_solution(self) -> tuple[QuantumCircuit, float]:
        """Returns the quantum circuit of the QAOA algorithm and the resulting solution.

        Returns:
            tuple: A tuple containing the QAOA circuit (QuantumCircuit) and the computed energy (float) of the solution.
        """
        circuit = self._qaoa_circuit_from_qubo()
        circuit = transpile(circuit, backend=self.backend)
        circuit.measure_all()
        state, counts, energy = evaluate_result(self.num_init, circuit, self.backend, self.qubo)

        return circuit, energy

    def _qaoa_circuit_from_qubo(self) -> QuantumCircuit:
        """
        Convert a QUBO matrix (NumPy array) into a QAOAAnsatz circuit.

        Args:
            Q_mat: (n, n) NumPy array representing the QUBO coefficients.
            reps: number of QAOA layers (the p parameter).

        Returns:
            A QAOAAnsatz instance implementing the cost Hamiltonian for the given QUBO.
        """
        Q_mat = self.qubo
        reps = self.layers
        n = Q_mat.shape[0]

        # Build binary QUBO dict
        # Keys of length 1 for diagonal, length 2 for off-diagonals
        h: dict[int, float] = {}
        J = {}
        for i in range(n):
            # diagonal term
            coeff = Q_mat[i, i]
            if coeff != 0:
                h[i] = h.get(i, 0.0) + coeff / 2
        for i in range(n):
            for j in range(i + 1, n):
                coeff = Q_mat[i, j] + Q_mat[j, i]  # ensure symmetry
                if coeff != 0:
                    J[(i, j)] = coeff / 4
                    h[i] = h.get(i, 0.0) + coeff / 4
                    h[j] = h.get(j, 0.0) + coeff / 4

        # Assemble Pauli terms
        pauli_list = []
        for i, val in h.items():
            label = ["I"] * n
            label[i] = "Z"
            pauli_list.append(("".join(label), val))
        for (i, j), val in J.items():
            label = ["I"] * n
            label[i] = "Z"
            label[j] = "Z"
            pauli_list.append(("".join(label), val))

        # Create cost operator
        cost_op = SparsePauliOp.from_list(pauli_list)

        return QAOAAnsatz(cost_operator=cost_op, reps=reps)


class VQE:
    def __init__(
        self, qubo: NDArray[np.float64], backend: BackendV2, ansatz: QuantumCircuit | None = None, num_init: int = 1
    ) -> None:
        self.qubo = qubo
        self.backend = backend
        self.num_init = num_init
        if ansatz is None:
            self.ansatz = EfficientSU2(num_qubits=qubo.shape[0], entanglement="linear")
        else:
            self.ansatz = ansatz

    def get_solution(self) -> tuple[QuantumCircuit, float]:
        """Returns the quantum circuit of the VQE algorithm and the resulting solution.
        Returns:
            tuple: A tuple containing the VQE circuit (QuantumCircuit) and the computed energy (float) of the solution.
        """
        circuit = self.ansatz
        circuit = transpile(circuit, backend=self.backend)
        circuit.measure_all()
        state, counts, energy = evaluate_result(self.num_init, circuit, self.backend, self.qubo)
        return circuit, energy
