from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.algorithms import MinimumEigensolverResult
    from qiskit_optimization import QuadraticProgram

from qiskit.algorithms.minimum_eigensolvers import QAOA as qiskitQAOA
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE as qiskitVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BackendSampler, Sampler
from qiskit.providers.fake_provider import FakeMontreal
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer, WarmStartQAOAOptimizer


def solve_using_w_qaoa(qubo: QuadraticProgram, noisy_flag: bool = False) -> MinimumEigensolverResult:
    if noisy_flag:
        wqaoa = W_QAOA(
            QAOA_params={"reps": 3, "optimizer": COBYLA(maxiter=100), "sampler": BackendSampler(FakeMontreal())}
        )
    else:
        wqaoa = W_QAOA(
            QAOA_params={
                "reps": 3,
                "optimizer": COBYLA(maxiter=100),
                "sampler": Sampler(),
            }
        )
    qc_wqaoa, res_wqaoa = wqaoa.get_solution(qubo)
    return res_wqaoa


def solve_using_qaoa(qubo: QuadraticProgram, noisy_flag: bool = False) -> Any:
    if noisy_flag:
        qaoa = QAOA(
            QAOA_params={"reps": 3, "optimizer": COBYLA(maxiter=100), "sampler": BackendSampler(FakeMontreal())}
        )
    else:
        qaoa = QAOA(
            QAOA_params={
                "reps": 3,
                "optimizer": COBYLA(maxiter=100),
                "sampler": Sampler(),
            }
        )
    qc_qaoa, res_qaoa = qaoa.get_solution(qubo)
    return res_qaoa


def solve_using_vqe(qubo: QuadraticProgram, noisy_flag: bool = False) -> Any:
    if noisy_flag:
        vqe = VQE(VQE_params={"optimizer": COBYLA(maxiter=100), "sampler": BackendSampler(FakeMontreal())})
    else:
        vqe = VQE(
            VQE_params={
                "optimizer": COBYLA(maxiter=100),
                "sampler": Sampler(),
                "ansatz": RealAmplitudes(num_qubits=qubo.get_num_binary_vars(), reps=3),
            }
        )
    qc_vqe, res_vqe = vqe.get_solution(qubo)
    return res_vqe


class VQE(qiskitVQE):  # type: ignore[misc]
    def __init__(self, VQE_params: dict[str, Any] | None = None) -> None:
        """Function which initializes the VQE class."""
        if VQE_params is None or type(VQE_params) is not dict:
            VQE_params = {}
        if VQE_params.get("optimizer") is None:
            VQE_params["optimizer"] = COBYLA(maxiter=1000)
        if VQE_params.get("sampler") is None:
            VQE_params["sampler"] = Sampler()
        if VQE_params.get("ansatz") is None:
            VQE_params["ansatz"] = RealAmplitudes()

        super().__init__(**VQE_params)

    def get_solution(self, qubo: QuadraticProgram) -> tuple[QuantumCircuit, MinimumEigensolverResult]:
        """Function which returns the quantum circuit of the VQE algorithm and the resulting solution."""
        vqe_result = MinimumEigenOptimizer(self).solve(qubo)
        qc = self.ansatz
        return qc, vqe_result


class QAOA(qiskitQAOA):  # type: ignore[misc]
    def __init__(self, QAOA_params: dict[str, Any] | None = None) -> None:
        """Function which initializes the QAOA class."""
        if QAOA_params is None or type(QAOA_params) is not dict:
            QAOA_params = {}
        if QAOA_params.get("optimizer") is None:
            QAOA_params["optimizer"] = COBYLA(maxiter=1000)
        if QAOA_params.get("reps") is None:
            QAOA_params["reps"] = 5
        if QAOA_params.get("sampler") is None:
            QAOA_params["sampler"] = Sampler()

        super().__init__(**QAOA_params)

    def get_solution(self, qubo: QuadraticProgram) -> tuple[QuantumCircuit, MinimumEigensolverResult]:
        """Function which returns the quantum circuit of the QAOA algorithm and the resulting solution."""
        qaoa_result = MinimumEigenOptimizer(self).solve(qubo)
        qc = self.ansatz
        return qc, qaoa_result


class W_QAOA:
    def __init__(self, W_QAOA_params: dict[str, Any] | None = None, QAOA_params: dict[str, Any] | None = None) -> None:
        """Function which initializes the QAOA class."""
        if type(W_QAOA_params) is not dict:
            W_QAOA_params = {}
        if W_QAOA_params.get("pre_solver") is None:
            W_QAOA_params["pre_solver"] = CobylaOptimizer()
        if W_QAOA_params.get("relax_for_pre_solver") is None:
            W_QAOA_params["relax_for_pre_solver"] = True
        if W_QAOA_params.get("qaoa") is None:
            if type(QAOA_params) is not dict:
                W_QAOA_params["qaoa"] = qiskitQAOA()
            else:
                W_QAOA_params["qaoa"] = qiskitQAOA(**QAOA_params)

        self.W_QAOA_params = W_QAOA_params
        self.qaoa = W_QAOA_params["qaoa"]

    def get_solution(self, qubo: QuadraticProgram) -> tuple[QuantumCircuit, MinimumEigensolverResult]:
        """Function which returns the quantum circuit of the W-QAOA algorithm and the resulting solution."""

        ws_qaoa = WarmStartQAOAOptimizer(**self.W_QAOA_params)
        res = ws_qaoa.solve(qubo)
        qc = self.W_QAOA_params["qaoa"].ansatz

        return qc, res
