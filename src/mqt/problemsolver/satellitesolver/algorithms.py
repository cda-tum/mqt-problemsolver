from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.providers import BackendV2

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz, efficient_su2
from qiskit.providers.fake_provider import GenericBackendV2
from scipy.optimize import minimize

from mqt.problemsolver.satellitesolver.utils import cost_op_from_qubo


def solve_using_qaoa(qubo: NDArray[np.float64], noisy_flag: bool = True, layers: int = 10, num_init: int = 5) -> float:
    backend = GenericBackendV2(noise_info=noisy_flag, num_qubits=qubo.shape[0])
    qaoa = QAOA(qubo, layers, num_init, backend)
    _qc_qaoa, res_qaoa = qaoa.get_solution()
    return res_qaoa


def solve_using_vqe(
    qubo: NDArray[np.float64],
    noisy_flag: bool = True,
    ansatz: QuantumCircuit | None = None,
    num_init: int = 5,
) -> float:
    backend = GenericBackendV2(noise_info=noisy_flag, num_qubits=qubo.shape[0])

    vqe = VQE(qubo, backend, ansatz, num_init)
    _, res_vqe = vqe.get_solution()
    return res_vqe


def bitstring_to_vector(bitstring: str) -> NDArray[np.int_]:
    """Convert a bitstring to a vector of integers."""
    return np.array([int(bit) for bit in bitstring])


def compute_expectation(counts: dict[str, int], qubo: NDArray[np.float64]) -> float:
    """Computes the expectation value of the quadratic form Q with respect to the given counts.

    Args:
        counts: A dictionary where keys are bitstrings and values are their counts.
        qubo: The QUBO matrix represented as a NumPy array.

    Returns:
    -------
    float
        The expectation value of the quadratic form Q with respect to the counts.
    """
    total_counts = sum(counts.values())
    expectation = 0.0
    for bitstring, count in counts.items():
        x = bitstring_to_vector(bitstring)
        energy = x @ qubo @ x  # Compute x^T Q x
        weight = count / total_counts
        expectation += weight * energy

    return expectation


def cost_func(params: list[float], circuit: QuantumCircuit, backend: BackendV2, qubo: NDArray[np.float64]) -> float:
    """Computes the cost function for QAOA optimization.

    Args:
        params: A list of floats representing the parameters for the QAOA circuit
        circuit: The QuantumCircuit representing the QAOA circuit to be evaluated.
        backend: The backend to run the circuit on.
        qubo: The QUBO matrix representing the problem.

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
    num_init: int,
    circuit: QuantumCircuit,
    backend: BackendV2,
    qubo: NDArray[np.float64],
) -> tuple[list[float], QuantumCircuit]:
    """Optimizes the QAOA parameters using multiple random initializations.

    Args:
        num_init: The number of random initializations to perform.
        circuit: The QuantumCircuit representing the QAOA circuit to be optimized.
        backend: The backend to run the circuit on.
        qubo: The QUBO matrix representing the problem.

    Returns:
        A list of floats representing the optimized parameters that minimize the cost function.
    """
    best_cost = float("inf")
    best_params = []

    rng = np.random.default_rng(42)
    for _ in range(num_init):
        params = rng.uniform(0, 2 * np.pi, size=len(circuit.parameters))
        res = minimize(
            cost_func,
            x0=params,
            args=(circuit, backend, qubo),
            method="COBYLA",
            bounds=[(0, 2 * np.pi) for _ in range(len(circuit.parameters))],
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
    num_init: int,
    circuit: QuantumCircuit,
    backend: BackendV2,
    qubo: NDArray[np.float64],
) -> tuple[str, int, float]:
    """Evaluates the result of the QAOA circuit.

    Args:
        num_init: The number of initializations to perform.
        circuit: The QAOA circuit to be evaluated.
        backend: The backend to run the circuit on.
        qubo: The QUBO matrix representing the problem.

    Returns:
        A tuple containing:
        - The found state as a bitstring.
        - The count of that state.
        - The energy computed from the QUBO matrix.
    """
    _optimized_params, circuit = optimize_with_multiple_init_parameters(num_init, circuit, backend, qubo)

    job = backend.run(circuit, shots=10000)
    result = job.result()
    counts = result.get_counts()

    found_state = max(counts, key=counts.get)
    x = bitstring_to_vector(found_state)
    energy = x @ qubo @ x

    print("Found Energy:", energy)

    return found_state, counts[found_state], energy


class QAOA:
    def __init__(
        self, qubo: NDArray[np.float64], layers: int, num_init: int, backend: BackendV2, reps: int = 2
    ) -> None:
        self.qubo = qubo
        self.layers = layers
        self.num_init = num_init
        self.backend = backend
        self.reps = reps

    def get_solution(self) -> tuple[QuantumCircuit, float]:
        """Returns the quantum circuit of the QAOA algorithm and the resulting solution.

        Returns:
            tuple: A tuple containing the QAOA circuit (QuantumCircuit) and the computed energy (float) of the solution.
        """
        circuit = self._qaoa_circuit_from_cost_op()
        circuit.measure_all()
        circuit = transpile(circuit, backend=self.backend)
        _state, _counts, energy = evaluate_result(self.num_init, circuit, self.backend, self.qubo)

        return circuit, energy

    def _qaoa_circuit_from_cost_op(self) -> QuantumCircuit:
        """Creates a QAOA circuit from a cost operator."""
        cost_op, _ = cost_op_from_qubo(self.qubo)
        return QAOAAnsatz(cost_operator=cost_op, reps=self.reps)


class VQE:
    def __init__(
        self, qubo: NDArray[np.float64], backend: BackendV2, ansatz: QuantumCircuit | None = None, num_init: int = 1
    ) -> None:
        self.qubo = qubo
        self.backend = backend
        self.num_init = num_init
        if ansatz is None:
            self.ansatz = efficient_su2(num_qubits=qubo.shape[0], entanglement="linear")
        else:
            self.ansatz = ansatz

    def get_solution(self) -> tuple[QuantumCircuit, float]:
        """Returns the quantum circuit of the VQE algorithm and the resulting solution.

        Returns:
            tuple: A tuple containing the VQE circuit (QuantumCircuit) and the computed energy (float) of the solution.
        """
        circuit = self.ansatz
        circuit.measure_all()
        circuit = transpile(circuit, backend=self.backend)
        _state, _counts, energy = evaluate_result(self.num_init, circuit, self.backend, self.qubo)
        return circuit, energy
