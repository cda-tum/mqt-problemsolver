import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters.quadratic_program_to_qubo import (
    QuadraticProgramToQubo,
)
from qiskit_optimization.translators import from_docplex_mp

from mqt.problemsolver.AcquisitionRequest import (
    ORBIT_DURATION,
    R_E,
    R_S,
    ROTATION_SPEED_SATELLITE,
    AcquisitionRequest,
)

from qiskit_optimization.algorithms import CobylaOptimizer, WarmStartQAOAOptimizer
from qiskit import Aer
from qiskit.algorithms import QAOA as qiskitQAOA
from qiskit.algorithms.optimizers import L_BFGS_B

def init_random_acquisition_requests(n: int) -> list[AcquisitionRequest]:
    """Returns list of n random acquisition requests"""
    acquisition_requests = []
    for _ in range(n):
        acquisition_requests.append(
            AcquisitionRequest(
                create_acquisition_position(), 1, np.random.randint(1, 3)
            )
        )

    return sort_acquisition_requests(acquisition_requests)


def sample_most_likely(state_vector: dict) -> str:
    values = list(state_vector.values())
    k = np.argmax(np.abs(values))
    return list(state_vector.keys())[k]


def get_satellite_position(t):
    # Return position of the satellite as a vector
    longitude = 2 * np.pi / ORBIT_DURATION * t
    return R_S * np.array([np.cos(longitude), np.sin(longitude), 0])


def create_acquisition_position(longitude=None, latitude=None):
    # Returns random position of acquisition close to the equator as vector
    if longitude is None:
        longitude = 2 * np.pi * np.random.rand()
    if latitude is None:
        latitude = np.random.uniform(
            np.pi / 2 - 15 / 360 * 2 * np.pi, np.pi / 2 + 15 / 360 * 2 * np.pi
        )

    return R_E * np.array(
        [
            np.cos(longitude) * np.sin(latitude),
            np.sin(longitude) * np.sin(latitude),
            np.cos(latitude),
        ]
    )


def calc_needed_time_between_acquisition_attempts(
    first_acq: AcquisitionRequest, second_acq: AcquisitionRequest
):
    # Calculates the time needed for the satellite to change its focus from one acquisition
    # (first_acq) to the other (second_acq)
    # Assumption: required position of the satellite is constant over possible imaging attempts
    delta_r1 = first_acq.position - first_acq.get_average_satellite_position()
    delta_r2 = second_acq.position - second_acq.get_average_satellite_position()
    theta = np.arccos(
        delta_r1 @ delta_r2 / (np.linalg.norm(delta_r1) * np.linalg.norm(delta_r2))
    )

    return theta / (ROTATION_SPEED_SATELLITE * 2 * np.pi)


def transition_possible(acq_1: AcquisitionRequest, acq_2: AcquisitionRequest) -> bool:
    """Returns True if transition between acq_1 and acq_2 is possible, False otherwise"""
    t_maneuver = calc_needed_time_between_acquisition_attempts(acq_1, acq_2)
    t1 = np.mean(acq_1.get_imaging_attempts())
    t2 = np.mean(acq_2.get_imaging_attempts())
    if t1 < t2:
        return t2 - t1 > t_maneuver + acq_1.duration
    elif t2 < t1:
        return t1 - t2 > t_maneuver + acq_2.duration
    elif t1 == t2:
        return False


def get_transition_possibility_matrix(acqs: list):
    # Returns a matrix with boolean entries if a transition between two acquisitions is possible for
    # all possible combinations of acquisitions
    possibility_matrix = np.zeros((len(acqs), len(acqs)))
    for i in range(len(acqs)):
        for j in range(len(acqs) - i):
            possibility_matrix[i, j + i] += float(
                transition_possible(acqs[i], acqs[i + j])
            )

    return possibility_matrix


def sort_acquisition_requests(acqs: list):
    # Sorts acquisition requests in order of ascending longitudes
    longitudes = np.zeros(len(acqs))
    acqs_sorted = []
    for idx, acq in enumerate(acqs):
        longitudes[idx] += acq.get_longitude_angle()
    indices_sorted = np.argsort(longitudes)
    for i in indices_sorted:
        acqs_sorted.append(acqs[i])

    return acqs_sorted


def plot_acqisition_requests(acqs: list):
    # Plots all acquisition requests on a sphere
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
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

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def create_satellite_doxplex(all_acqs: list) -> Model:
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
                mdl.add_constraint(
                    (requests[i] + requests[j])<= 1
                )

    # Add objective function
    mdl.minimize(-mdl.sum(requests[i] * values[i] for i in range(len(all_acqs))))
    return mdl


def convert_docplex_to_qubo(
    model, penalty=None
) -> Tuple[QuadraticProgramToQubo, QuadraticProgram]:
    """Converts a docplex model to a qubo"""
    qp = from_docplex_mp(model)
    conv = QuadraticProgramToQubo(penalty=penalty)
    return conv, conv.convert(qp)


def get_longitude(vector):
    temp = vector * np.array([1, 1, 0])
    temp /= np.linalg.norm(temp)
    if temp[1] >= 0:
        long = np.arccos(temp[0])
    else:
        long = 2 * np.pi - np.arccos(temp[0])
    return long


def save_coordinates_to_file(acqs, format="coordinates"):
    cwd = os.getcwd()
    dir = os.path.join(cwd, "test_coordinates")
    idx = len([i for i in os.listdir(dir) if "coordinates" in i])

    with open(os.path.join(dir, "coordinates_" + str(idx) + ".txt"), "w") as f:
        for i in range(len(acqs)):
            if format == "coordinates":
                long, lat = acqs[i].get_coordinates()
            elif format == "angles":
                long, lat = str(acqs[i].get_longitude_angle()), str(
                    acqs[i].get_latitude_angle()
                )
            f.write(long + "  ;  " + lat + "\n")


def check_solution(
    ac_reqs: list[AcquisitionRequest], solution_vector: list[int]
) -> bool:
    """Checks if the determined solution is valid and does not violate any constraints."""
    for i in range(len(ac_reqs) - 1):
        for j in range(i + 1, len(ac_reqs)):
            if (
                solution_vector[i] + solution_vector[j] == 2
            ) and not transition_possible(ac_reqs[i], ac_reqs[j]):
                return False
    return True


def calc_sol_value(ac_reqs: list[AcquisitionRequest], solution_vector: list[int]):
    """Calculates the value of the solution vector"""
    value = 0
    for i in range(len(ac_reqs)):
        value += solution_vector[i] * ac_reqs[i].imaging_attempt_score
    return value


class QAOA(qiskitQAOA):
    def __init__(self, QAOA_params=None):
        """Function which initializes the QAOA class."""
        if QAOA_params is None or not type(QAOA_params) is dict:
            QAOA_params = dict()
        if QAOA_params.get("optimizer") is None:
            print("Optimizer not specified, using L-BFGS-B.")
            QAOA_params["optimizer"] = L_BFGS_B(maxiter=10000)
        if QAOA_params.get("reps") is None:
            QAOA_params["reps"] = 5
        if QAOA_params.get("quantum_instance") is None:
            QAOA_params["quantum_instance"] = Aer.get_backend("qasm_simulator")

        super().__init__(**QAOA_params)

    def get_solution(self, qubo, aux_operators=None):
        """Function which returns the quantum circuit of the QAOA algorithm and the resulting solution."""
        operator, offset = qubo.to_ising()
        qaoa_result = self.compute_minimum_eigenvalue(
            operator, aux_operators=aux_operators
        )
        qc = self.ansatz.bind_parameters(qaoa_result.optimal_point)
        return qc, qaoa_result






class W_QAOA:
    def __init__(self, W_QAOA_params=None, QAOA_params=None):
        """Function which initializes the QAOA class."""
        if not type(W_QAOA_params) is dict:
            W_QAOA_params = dict()
        if W_QAOA_params.get("pre_solver") is None:
            W_QAOA_params["pre_solver"] = CobylaOptimizer()
        if W_QAOA_params.get("relax_for_pre_solver") is None:
            W_QAOA_params["relax_for_pre_solver"] = True
        if W_QAOA_params.get("qaoa") is None:
            if not type(QAOA_params) is dict:
                W_QAOA_params["qaoa"] = QAOA.QAOA()
            else:
                W_QAOA_params["qaoa"] = QAOA.QAOA(**QAOA_params)

        self.W_QAOA_params = W_QAOA_params
        self.qaoa = W_QAOA_params["qaoa"]

    def get_solution(self, qubo):
        """Function which returns the quantum circuit of the W-QAOA algorithm and the resulting solution."""

        ws_qaoa = WarmStartQAOAOptimizer(**self.W_QAOA_params)
        res = ws_qaoa.solve(qubo)
        qc = self.W_QAOA_params["qaoa"].ansatz

        return qc, res
