from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.problemsolver.satellitesolver import algorithms, utils
from mqt.problemsolver.satellitesolver.evaluator import eval_all_instances_satellite_solver
from mqt.problemsolver.satellitesolver.imaging_location import LocationRequest

if TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng(42)  # Set seed for reproducibility


@pytest.fixture
def qubo() -> NDArray[np.float64]:
    ac_reqs = utils.init_random_location_requests(3, rng=rng)
    return utils.create_satellite_qubo(ac_reqs)


def test_solve_using_qaoa(qubo: NDArray[np.float64]) -> None:
    res_qaoa = algorithms.solve_using_qaoa(qubo)
    assert res_qaoa is not None


def test_solve_using_vqe(qubo: NDArray[np.float64]) -> None:
    res_qaoa = algorithms.solve_using_vqe(qubo)
    assert res_qaoa is not None


def test_eval_all_instances_satellite_solver() -> None:
    eval_all_instances_satellite_solver(min_qubits=3, max_qubits=4, stepsize=1, num_runs=1)


def test_init_random_acquisition_requests() -> None:
    req = utils.init_random_location_requests(5, rng=rng)
    assert len(req) == 5
    assert isinstance(req[0], LocationRequest)


def test_location_request() -> None:
    req = LocationRequest(np.array([1.5, 1.5, 1.5]), 5)
    assert req.imaging_attempt_score == 5
    assert req.get_longitude_angle()
    assert req.get_latitude_angle()
    assert req.get_coordinates()
