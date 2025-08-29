from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvalsh
from qiskit.quantum_info import Pauli, SparsePauliOp

from mqt.problemsolver.satellite_solver.imaging_location import (
    R_E,
    R_S,
    ROTATION_SPEED_SATELLITE,
    LocationRequest,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def init_random_location_requests(n: int, rng: np.random.Generator | None = None) -> list[LocationRequest]:
    """Returns list of n random acquisition requests."""
    if rng is None:
        rng = np.random.default_rng(10)

    acquisition_requests = [
        LocationRequest(position=create_acquisition_position(rng=rng), imaging_attempt_score=float(rng.integers(1, 3)))
        for _ in range(n)
    ]

    return sort_acquisition_requests(acquisition_requests)


def get_success_ratio(ac_reqs: list[LocationRequest], qubo: NDArray[np.float64], solution_vector: list[int]) -> float:
    exact_result = solve_classically(qubo)
    # sum over all LocationRequests and sum over their imaging_attempt_score if the respective indicator in sol[index] is 1
    solution_vector = solution_vector[::-1]
    return (
        sum(-ac_req.imaging_attempt_score for index, ac_req in enumerate(ac_reqs) if solution_vector[index] == 1)
        / exact_result
    )


def create_acquisition_position(
    longitude: float | None = None,
    latitude: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    # Returns random position of acquisition close to the equator as vector
    if rng is None:
        rng = np.random.default_rng()
    if longitude is None:
        longitude = 2 * np.pi * rng.random()
    if latitude is None:
        latitude = rng.uniform(np.pi / 2 - 15 / 360 * 2 * np.pi, np.pi / 2 + 15 / 360 * 2 * np.pi)

    res = R_E * np.array([
        np.cos(longitude) * np.sin(latitude),
        np.sin(longitude) * np.sin(latitude),
        np.cos(latitude),
    ])
    return res.astype(np.float64)


def calc_needed_time_between_acquisition_attempts(
    first_acq: LocationRequest, second_acq: LocationRequest
) -> NDArray[np.float64]:
    # Calculates the time needed for the satellite to change its focus from one acquisition
    # (first_acq) to the other (second_acq)
    # Assumption: required position of the satellite is constant over possible imaging attempts
    delta_r1: NDArray[np.float64] = first_acq.position - first_acq.get_average_satellite_position()
    delta_r2: NDArray[np.float64] = second_acq.position - second_acq.get_average_satellite_position()
    theta = np.arccos(delta_r1 @ delta_r2 / (np.linalg.norm(delta_r1) * np.linalg.norm(delta_r2)))
    result = theta / (ROTATION_SPEED_SATELLITE * 2 * np.pi)

    return cast("NDArray[np.float64]", result)


def transition_possible(acq_1: LocationRequest, acq_2: LocationRequest) -> bool:
    """Returns True if transition between acq_1 and acq_2 is possible, False otherwise."""
    t_maneuver = cast("float", calc_needed_time_between_acquisition_attempts(acq_1, acq_2))
    t1 = acq_1.imaging_attempt
    t2 = acq_2.imaging_attempt
    if t1 < t2:
        return (t2 - t1) > t_maneuver
    if t2 < t1:
        return (t1 - t2) > t_maneuver
    if t1 == t2:
        return False
    return False


def sort_acquisition_requests(acqs: list[LocationRequest]) -> list[LocationRequest]:
    # Sorts acquisition requests in order of ascending longitudes
    longitudes = np.zeros(len(acqs))
    acqs_sorted = []
    for idx, acq in enumerate(acqs):
        longitudes[idx] += acq.get_longitude_angle()
    indices_sorted = np.argsort(longitudes)
    for i in indices_sorted:
        acqs_sorted.append(acqs[i])  # noqa: PERF401

    return acqs_sorted


def plot_acquisition_requests(acqs: list[LocationRequest]) -> None:
    # Plots all acquisition requests on a sphere
    phi, theta = np.mgrid[0 : np.pi : 100j, 0 : 2 * np.pi : 100j]  # type: ignore[misc]
    x = R_E * np.sin(phi) * np.cos(theta)
    y = R_E * np.sin(phi) * np.sin(theta)
    z = R_E * np.cos(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="c", alpha=0.6, linewidth=0)
    ax.plot(
        R_S * np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / 10000)),
        R_S * np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / 10000)),
        np.zeros(10000),
    )

    for i in range(len(acqs)):
        xi, yi, zi = acqs[i].position[0], acqs[i].position[1], acqs[i].position[2]
        ax.scatter(xi, yi, zi, color="k", s=20)

    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()


def sample_most_likely(state_vector: dict[str, int]) -> list[int]:
    values = list(state_vector.values())
    k = np.argmax(np.abs(values))
    res = list(state_vector.keys())[k]
    # convert str of binary values to list of int
    return [int(x) for x in res]


def check_solution(ac_reqs: list[LocationRequest], solution_vector: list[int]) -> bool:
    """Checks if the determined solution is valid and does not violate any constraints."""
    solution_vector = solution_vector[::-1]
    for i in range(len(ac_reqs) - 1):
        for j in range(i + 1, len(ac_reqs)):
            if (solution_vector[i] + solution_vector[j] == 2) and not transition_possible(ac_reqs[i], ac_reqs[j]):
                return False
    return True


def create_satellite_qubo(all_acqs: list[LocationRequest], penalty: int = 8) -> NDArray[np.float64]:
    """Creates a QUBO matrix directly for the satellite location request problem.

    Args:
        all_acqs: List of all acquisition requests.
        penalty: Penalty for conflicting requests. Defaults to 8.

    Returns:
        A QUBO object representing the problem.
    """
    n = len(all_acqs)
    values = [req.imaging_attempt_score for req in all_acqs]

    # Initialize QUBO matrix
    q = np.zeros((n, n), dtype=float)

    # Objective (diagonal) terms
    for i, v in enumerate(values):
        q[i, i] = -v

    # Penalty for conflicts (off-diagonals)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not transition_possible(all_acqs[i], all_acqs[j]):
                q[i, j] = penalty
                q[j, i] = penalty  # ensure symmetry

    return q


def cost_op_from_qubo(q: NDArray[np.float64]) -> tuple[SparsePauliOp, float]:
    """Convert a QUBO matrix to an Ising Hamiltonian.

    Args:
        q: QUBO matrix (symmetric, square numpy array).

    Returns:
        A tuple (qubit_op, offset) representing the Ising Hamiltonian and constant offset.
    """
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        msg = "QUBO matrix must be a square numpy array."
        raise ValueError(msg)

    num_vars = q.shape[0]
    zero = np.zeros(num_vars, dtype=bool)
    pauli_list = []
    offset = 0.0

    for i in range(num_vars):
        # Diagonal: linear terms
        coef = q[i, i]
        weight = coef / 2
        z_p = zero.copy()
        z_p[i] = True
        pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))
        offset += weight

        for j in range(i + 1, num_vars):
            coef = q[i, j] + q[j, i]  # ensure symmetry
            if coef == 0:
                continue

            weight = coef / 4
            # z_i z_j term
            z_p = zero.copy()
            z_p[i] = True
            z_p[j] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), weight))

            # local z_i term
            z_p = zero.copy()
            z_p[i] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))

            # local z_j term
            z_p = zero.copy()
            z_p[j] = True
            pauli_list.append(SparsePauliOp(Pauli((z_p, zero)), -weight))

            offset += weight

    qubit_op = sum(pauli_list).simplify(atol=0) if pauli_list else SparsePauliOp("I" * max(1, num_vars), 0)

    return qubit_op, offset


def solve_classically(qubo: NDArray[np.float64]) -> float:
    """Solve the Hamiltonian problem classically using eigenvalue decomposition.

    Args:
        qubo: The Hamiltonian matrix derived from the QUBO.

    Returns:
        The minimum eigenvalue of the Hamiltonian matrix.
    """
    h, offset = cost_op_from_qubo(qubo)
    h_mat = h.to_matrix()
    eigenvalues = eigvalsh(h_mat)

    # Find the minimum eigenvalue
    return float(np.min(eigenvalues) + offset)


def get_longitude(vector: NDArray[np.float64]) -> float:
    temp = vector * np.array([1, 1, 0])
    temp /= np.linalg.norm(temp)
    return cast("float", np.arccos(temp[0]) if temp[1] >= 0 else 2 * np.pi - np.arccos(temp[0]))
