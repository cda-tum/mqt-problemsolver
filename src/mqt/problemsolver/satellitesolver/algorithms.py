from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2  # type: ignore[import-not-found]
from scipy.optimize import minimize


def solve_using_qaoa(qubo: NDArray[np.float64], noisy_flag: bool = False, layers: int = 10, num_init: int = 5) -> Any:
    fake_backend = FakeMontrealV2()

    if noisy_flag:
        backend = AerSimulator.from_backend(fake_backend)
        assert backend.options.get("noise_model") is not None
    else:
        backend = AerSimulator.from_backend(fake_backend, noise_model=None)
        assert backend.options.get("noise_model") is None

    backend.set_max_qubits(fake_backend.configuration().num_qubits)

    qaoa = QAOA(qubo, layers, num_init, backend)
    qc_qaoa, res_qaoa = qaoa.get_solution()
    return res_qaoa


# def solve_using_vqe(qubo: QuadraticProgram, noisy_flag: bool = False) -> Any:
#     if noisy_flag:
#         vqe = VQE(VQE_params={"optimizer": COBYLA(maxiter=100), "sampler": BackendSampler(FakeMontreal())})
#     else:
#         vqe = VQE(
#             VQE_params={
#                 "optimizer": COBYLA(maxiter=100),
#                 "sampler": Sampler(),
#                 "ansatz": RealAmplitudes(num_qubits=qubo.get_num_binary_vars(), reps=3),
#             }
#         )
#     qc_vqe, res_vqe = vqe.get_solution(qubo)
#     return res_vqe


# def solve_using_w_qaoa(qubo: QuadraticProgram, noisy_flag: bool = False) -> MinimumEigensolverResult:
#     if noisy_flag:
#         wqaoa = W_QAOA(
#             QAOA_params={"reps": 3, "optimizer": COBYLA(maxiter=100), "sampler": BackendSampler(FakeMontreal())}
#         )
#     else:
#         wqaoa = W_QAOA(
#             QAOA_params={
#                 "reps": 3,
#                 "optimizer": COBYLA(maxiter=100),
#                 "sampler": Sampler(),
#             }
#         )
#     qc_wqaoa, res_wqaoa = wqaoa.get_solution(qubo)
#     return res_wqaoa


def bitstring_to_vector(bitstring: str) -> NDArray[np.int_]:
    """Convert a bitstring to a vector of integers."""
    return np.array([int(bit) for bit in bitstring])


def compute_expectation(counts: dict[str, int], Q: NDArray[np.float64]) -> float:
    """Compute the expectation value of the quadratic form Q with respect to the given counts.

    Parameters
    ----------
    counts : dict[str, int]
        A dictionary where keys are bitstrings and values are their counts.
    Q : NDArray[np.float64]
        The QUBO matrix represented as a NumPy array.
    
    Returns
    -------
    float
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


# class VQE(qiskitVQE):  # type: ignore[misc]
#     def __init__(self, VQE_params: dict[str, Any] | None = None) -> None:
#         """Function which initializes the VQE class."""
#         if VQE_params is None or not isinstance(VQE_params, dict):
#             VQE_params = {}
#         if VQE_params.get("optimizer") is None:
#             VQE_params["optimizer"] = COBYLA(maxiter=1000)
#         if VQE_params.get("sampler") is None:
#             VQE_params["sampler"] = Sampler()
#         if VQE_params.get("ansatz") is None:
#             VQE_params["ansatz"] = RealAmplitudes()

#         super().__init__(**VQE_params)

#     def get_solution(self, qubo: QuadraticProgram) -> tuple[QuantumCircuit, MinimumEigensolverResult]:
#         """Function which returns the quantum circuit of the VQE algorithm and the resulting solution."""
#         vqe_result = MinimumEigenOptimizer(self).solve(qubo)
#         qc = self.ansatz
#         return qc, vqe_result


class QAOA:
    def __init__(self, qubo: NDArray[np.float64], layers: int, num_init: int, backend: AerSimulator) -> None:
        self.qubo = qubo
        self.layers = layers
        self.num_init = num_init
        self.backend = backend

    def qaoa_circuit_from_qubo(self, include_barriers: bool) -> QuantumCircuit:
        """Constructs a QAOA circuit from the given QUBO matrix.

        Parameters
        ----------
        include_barriers : bool
            If True, adds barriers between layers in the circuit.

        Returns
        -------
        QuantumCircuit
            The QAOA circuit constructed from the QUBO matrix.
        """
        Q = self.qubo
        n_qubits = Q.shape[0]
        circuit = QuantumCircuit(n_qubits, n_qubits)
        gamma = Parameter("gamma")
        beta = Parameter("beta")
        interactions = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits) if Q[i, j] != 0]

        for i in range(n_qubits):
            circuit.h(i)

        for _ in range(self.layers):
            if include_barriers:
                circuit.barrier()
            for i, j in interactions:
                circuit.rzz(gamma, i, j)
            if include_barriers:
                circuit.barrier()
            for i in range(n_qubits):
                circuit.rx(beta, i)

        if include_barriers:
            circuit.barrier()
        for i in range(n_qubits):
            circuit.measure(i, i)

        return circuit

    def cost_func(self, params: list[float], circuit: QuantumCircuit) -> float:
        """Cost function for the QAOA optimization.

        Parameters
        ----------
        params : list[float]
            The parameters for the QAOA circuit, where params[0] is beta and params[1] is gamma.
        circuit : QuantumCircuit
            The QAOA circuit to be evaluated.
        
        Returns
        -------
        float
            The expectation value of the QUBO Hamiltonian with respect to the states produced by the circuit.
        """
        beta, gamma = params
        circuit = circuit.assign_parameters({circuit.parameters[0]: beta, circuit.parameters[1]: gamma})

        # Simulate the circuit and get the statevector
        transpiled_circuit = transpile(circuit, backend=self.backend)
        job = self.backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()

        return compute_expectation(counts, self.qubo)

    def optimize_with_multiple_init_parameters(self, circuit: QuantumCircuit) -> list[float]:
        """Optimize the QAOA parameters using multiple random initializations.

        Parameters
        ----------
        circuit : QuantumCircuit
            The QAOA circuit to be optimized.
        
        Returns
        -------
        list[float]
            The optimized parameters [beta, gamma] that minimize the cost function.
        """
        best_cost = float("inf")
        best_params = []

        for _ in range(self.num_init):
            print(_)
            beta, gamma = np.random.uniform(0, 2 * np.pi, size=2)
            res = minimize(
                self.cost_func,
                x0=[beta, gamma],
                args=(circuit),
                method="COBYLA",
                bounds=[(0, 2 * np.pi), (0, 2 * np.pi)],
                options={"disp": False, "maxiter": 1000},
            )

            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
        return best_params

    def evaluate_result(self, circuit: QuantumCircuit) -> tuple[str, int, float]:
        """Evaluate the result of the QAOA circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The QAOA circuit to be evaluated.
        
        Returns
        -------
        tuple[str, int, float]
            A tuple containing the found state as a bitstring, the count of that state, and the energy computed from the QUBO matrix.
        """
        optimized_params = self.optimize_with_multiple_init_parameters(circuit)
        optimized_beta, optimized_gamma = optimized_params

        circuit = circuit.assign_parameters(
            {circuit.parameters[0]: optimized_beta, circuit.parameters[1]: optimized_gamma}
        )

        transpiled_circuit = transpile(circuit, backend=self.backend)
        job = self.backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()

        found_state = max(counts, key=counts.get)
        x = bitstring_to_vector(found_state)
        energy = x @ self.qubo @ x  # Compute x^T Q x

        return found_state, counts[found_state], energy

    def get_solution(self) -> tuple[QuantumCircuit, float]:
        """Function which returns the quantum circuit of the QAOA algorithm and the resulting solution.

        Returns
        -------
        tuple[QuantumCircuit, float]
            A tuple containing the QAOA circuit and the computed energy of the solution.
        """
        circuit = self.qaoa_circuit_from_qubo(include_barriers=False)
        print(circuit.num_qubits)
        circuit = transpile(circuit, self.backend)

        state, counts, energy = self.evaluate_result(circuit)

        return circuit, energy


# class W_QAOA:
#     def __init__(self, W_QAOA_params: dict[str, Any] | None = None, QAOA_params: dict[str, Any] | None = None) -> None:
#         """Function which initializes the QAOA class."""

#         if not isinstance(W_QAOA_params, dict):
#             W_QAOA_params = {}
#         if W_QAOA_params.get("pre_solver") is None:
#             W_QAOA_params["pre_solver"] = CobylaOptimizer()
#         if W_QAOA_params.get("relax_for_pre_solver") is None:
#             W_QAOA_params["relax_for_pre_solver"] = True
#         if W_QAOA_params.get("qaoa") is None:
#             if not isinstance(QAOA_params, dict):
#                 W_QAOA_params["qaoa"] = qiskitQAOA()
#             else:
#                 W_QAOA_params["qaoa"] = qiskitQAOA(**QAOA_params)

#         self.W_QAOA_params = W_QAOA_params
#         self.qaoa = W_QAOA_params["qaoa"]

#     def get_solution(self, qubo: QuadraticProgram) -> tuple[QuantumCircuit, MinimumEigensolverResult]:
#         """Function which returns the quantum circuit of the W-QAOA algorithm and the resulting solution."""

#         ws_qaoa = WarmStartQAOAOptimizer(**self.W_QAOA_params)
#         res = ws_qaoa.solve(qubo)
#         qc = self.W_QAOA_params["qaoa"].ansatz

#         return qc, res
