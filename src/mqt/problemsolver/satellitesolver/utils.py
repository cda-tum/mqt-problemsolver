from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from docplex.mp.model import Model
from mqt.problemsolver.satellitesolver.ImagingLocation import (
    R_E,
    R_S,
    ROTATION_SPEED_SATELLITE,
    LocationRequest,
)

if TYPE_CHECKING:
    from qiskit_optimization import QuadraticProgram
import matplotlib.pyplot as plt
import numpy as np
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import (
    QuadraticProgramToQubo,
)
from qiskit_optimization.translators import from_docplex_mp


def init_random_acquisition_requests(n: int) -> list[LocationRequest]:
    """Returns list of n random acquisition requests"""
    np.random.seed(10)
    acquisition_requests = []
    for _ in range(n):
        acquisition_requests.append(
            LocationRequest(position=create_acquisition_position(), imaging_attempt_score=np.random.randint(1, 3))
        )

    return sort_acquisition_requests(acquisition_requests)


def get_success_ratio(ac_reqs: list[LocationRequest], qubo: QuadraticProgram, solution_vector: list[int]) -> float:
    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

    exact_mes = NumPyMinimumEigensolver()
    exact_result = MinimumEigenOptimizer(exact_mes).solve(qubo).fval
    # sum over all LocationRequests and sum over their imaging_attempt_score if the respective indicator in sol[index] is 1
    solution_vector = solution_vector[::-1]
    return cast(
        float,
        (
            sum(
                [
                    -ac_req.imaging_attempt_score
                    for ac_req, index in zip(ac_reqs, range(len(ac_reqs)))
                    if solution_vector[index] == 1
                ]
            )
            / exact_result
        ),
    )


def create_acquisition_position(
    longitude: float | None = None, latitude: float | None = None
) -> np.ndarray[Any, np.dtype[np.float64]]:
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
    return cast(np.ndarray[Any, np.dtype[np.float64]], res)


def calc_needed_time_between_acquisition_attempts(
    first_acq: LocationRequest, second_acq: LocationRequest
) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Calculates the time needed for the satellite to change its focus from one acquisition
    # (first_acq) to the other (second_acq)
    # Assumption: required position of the satellite is constant over possible imaging attempts
    delta_r1 = first_acq.position - first_acq.get_average_satellite_position()
    delta_r2 = second_acq.position - second_acq.get_average_satellite_position()
    theta = np.arccos(delta_r1 @ delta_r2 / (np.linalg.norm(delta_r1) * np.linalg.norm(delta_r2)))

    return theta / (ROTATION_SPEED_SATELLITE * 2 * np.pi)


def transition_possible(acq_1: LocationRequest, acq_2: LocationRequest) -> bool:
    """Returns True if transition between acq_1 and acq_2 is possible, False otherwise"""
    t_maneuver = cast(float, calc_needed_time_between_acquisition_attempts(acq_1, acq_2))
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
        acqs_sorted.append(acqs[i])

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


def create_satellite_doxplex(all_acqs: list[LocationRequest]) -> Model:
    """Returns a doxplex model for the satellite problem"""
    mdl = Model("satellite model")
    # Create binary variables for each acquisition request
    requests = mdl.binary_var_list(len(all_acqs), name="requests")
    values = []
    for req in all_acqs:
        values.append(req.imaging_attempt_score)

    # Add constraints for each acquisition request
    for i in range(len(all_acqs) - 1):
        for j in range(i + 1, len(all_acqs)):
            if not transition_possible(all_acqs[i], all_acqs[j]):
                mdl.add_constraint((requests[i] + requests[j]) <= 1)

    # Add objective function
    mdl.minimize(-mdl.sum(requests[i] * values[i] for i in range(len(all_acqs))))
    return mdl


def convert_docplex_to_qubo(
    model: Model, penalty: int | None = None
) -> tuple[QuadraticProgramToQubo, QuadraticProgram]:
    """Converts a docplex model to a qubo"""
    qp = from_docplex_mp(model)
    conv = QuadraticProgramToQubo(penalty=penalty)
    return conv, conv.convert(qp)


def get_longitude(vector: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    temp = vector * np.array([1, 1, 0])
    temp /= np.linalg.norm(temp)
    return cast(float, np.arccos(temp[0]) if temp[1] >= 0 else 2 * np.pi - np.arccos(temp[0]))
