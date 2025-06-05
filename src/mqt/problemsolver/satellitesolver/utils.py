from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np
from qubovert import QUBO  # type: ignore[import-not-found]

from mqt.problemsolver.satellitesolver.ImagingLocation import (
    R_E,
    R_S,
    ROTATION_SPEED_SATELLITE,
    LocationRequest,
)


def init_random_location_requests(n: int) -> list[LocationRequest]:
    """Returns list of n random acquisition requests"""
    np.random.seed(10)
    acquisition_requests = [
        LocationRequest(position=create_acquisition_position(), imaging_attempt_score=np.random.randint(1, 3))
        for _ in range(n)
    ]

    return sort_acquisition_requests(acquisition_requests)


def get_success_ratio(ac_reqs: list[LocationRequest], qubo: NDArray[np.float64], solution_vector: list[int]) -> float:
    exact_result = solve_with_classical_brute_force(qubo)
    # sum over all LocationRequests and sum over their imaging_attempt_score if the respective indicator in sol[index] is 1
    solution_vector = solution_vector[::-1]
    return (
        sum(
            [
                -ac_req.imaging_attempt_score
                for ac_req, index in zip(ac_reqs, range(len(ac_reqs)))
                if solution_vector[index] == 1
            ]
        )
        / exact_result
    )


def create_acquisition_position(longitude: float | None = None, latitude: float | None = None) -> NDArray[np.float64]:
    # Returns random position of acquisition close to the equator as vector
    if longitude is None:
        longitude = 2 * np.pi * np.random.rand()
    if latitude is None:
        latitude = np.random.uniform(np.pi / 2 - 15 / 360 * 2 * np.pi, np.pi / 2 + 15 / 360 * 2 * np.pi)

    res = R_E * np.array(
        [
            np.cos(longitude) * np.sin(latitude),
            np.sin(longitude) * np.sin(latitude),
            np.cos(latitude),
        ]
    )
    return cast("NDArray[np.float64]", res)


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
    """Returns True if transition between acq_1 and acq_2 is possible, False otherwise"""
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


def plot_acqisition_requests(acqs: list[LocationRequest]) -> None:
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


def create_satellite_qubo(all_acqs: list[LocationRequest], penalty: int = 8) -> QUBO:
    """Creates a QUBO matrix directly for the satellite location request problem

    Parameters
    ----------
    all_acqs : list[LocationRequest]
        List of all acquisition requests.
    penalty : int, optional
        Penalty for conflicting requests, by default 8

    Returns
    -------
    QUBO: QUBO
        A QUBO object representing the problem.
    """
    n = len(all_acqs)
    Q_dict = {}
    values = [req.imaging_attempt_score for req in all_acqs]

    # Add objective function contributions (diagonal terms)
    for i in range(n):
        Q_dict[(i, i)] = -values[i]

    # Add penalty terms for conflicting requests (off-diagonal terms)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not transition_possible(all_acqs[i], all_acqs[j]):
                Q_dict[(i, j)] = penalty

    # Create the QUBO object
    return QUBO(Q_dict)


def qubo_to_matrix(qubo: QUBO) -> NDArray[np.float64]:
    """Extracts a NumPy matrix from a qubovert QUBO object

    Parameters
    ----------
    qubo : QUBO
        A QUBO object representing the problem.

    Returns
    -------
    NDArray[np.float64]
        A NumPy array representing the QUBO matrix.
    """
    Q = np.zeros((len(qubo.variables), len(qubo.variables)))
    for key, value in qubo.items():
        # Handle diagonal and off-diagonal separately
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            Q[i, j] = value
            if i != j:
                Q[j, i] = value  # Ensure symmetry
        elif isinstance(key, tuple) and len(key) == 1:
            i = key[0]
            Q[i, i] = value
        else:
            msg = f"Unexpected key format: {key}"
            raise ValueError(msg)
    return Q


def solve_with_classical_brute_force(qubo: NDArray[np.float64]) -> float:
    """Solves the QUBO problem using a brute-force approach.
    This function iterates over all possible binary solutions and calculates the energy for each solution.

    Parameters
    ----------
    qubo : NDArray[np.float64]
        A NumPy array representing the QUBO matrix.

    Returns
    -------
    float
        The minimum energy found across all solutions.
    """
    n = np.shape(qubo)[0]
    min_energy = float("inf")

    # Iterate over all possible solutions
    for solution in itertools.product([0, 1], repeat=n):
        energy = sum(qubo[i, j] * solution[i] * solution[j] for i in range(n) for j in range(n))
        if energy < min_energy:
            min_energy = energy
    return min_energy


def get_longitude(vector: NDArray[np.float64]) -> float:
    temp = vector * np.array([1, 1, 0])
    temp /= np.linalg.norm(temp)
    return cast("float", np.arccos(temp[0]) if temp[1] >= 0 else 2 * np.pi - np.arccos(temp[0]))
