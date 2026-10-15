# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module tests the generation of QAOA circuits (both the standard and heavy-hex optimized variants)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import qiskit
import sympy as sp
from qiskit.quantum_info import Operator

from mqt.problemsolver.qubo_tools import Calibration, QuboGenerator

if TYPE_CHECKING:
    import numpy.typing as npt


@pytest.fixture
def simple_generator() -> QuboGenerator:
    """Provides a simple QuboGenerator instance for testing.

    Returns:
        QuboGenerator: An instance of QuboGenerator with the objective function x1 * x2 * x3 * x4 * x5.
    """
    symbols = [sp.Symbol(f"x_{i + 1}") for i in range(5)]
    obj = sp.Mul(*symbols)
    generator = QuboGenerator(obj)
    generator.disable_caching = True
    return generator


@pytest.fixture
def simple_device() -> Calibration:
    """Provides a test device calibration with heavy-hex-style connectivity."""
    heavy_hex_coupling = []

    def get_qubit_index(row: int, cell: int, off: bool) -> int:
        """Converts a (row, cell, off) tuple to a qubit index.

        Args:
            row (int): The row of the qubit.
            cell (int): The cell of the qubit.
            off (bool): Whether the qubit is in a main row or an off row.

        Returns:
            int: The qubit index.
        """
        if not off:
            return 19 * row + cell
        return 19 * row + 15 + cell

    for row in range(7):
        for cell in range(15):
            current = get_qubit_index(row, cell, False)
            if cell != 14:
                neighbor = get_qubit_index(row, cell + 1, False)
                heavy_hex_coupling.append((current, neighbor))
            if cell % 4 == 0:
                if row % 2 == 0 and row != 6:
                    below = get_qubit_index(row, cell // 4, True)
                    heavy_hex_coupling.append((current, below))
                elif row % 2 == 1:
                    above = get_qubit_index(row - 1, cell // 4, True)
                    heavy_hex_coupling.append((current, above))
            if cell % 4 == 2:
                if row % 2 == 0 and row != 0:
                    above = get_qubit_index(row - 1, cell // 4, True)
                    heavy_hex_coupling.append((current, above))
                elif row % 2 == 1:
                    below = get_qubit_index(row, cell // 4, True)
                    heavy_hex_coupling.append((current, below))

    longest_hamiltonian_path = []
    for row in range(7):
        for cell in range(15):
            if row == 0 and cell == 0:
                continue  # we skip the first qubit.
            current = get_qubit_index(row, cell if row % 2 == 1 else (14 - cell), False)
            longest_hamiltonian_path.append(current)
        if row != 6:
            longest_hamiltonian_path.append(get_qubit_index(row, 3 if row % 2 == 1 else 0, True))

    num_qubits = max(max(pair) for pair in heavy_hex_coupling) + 1

    return Calibration(
        num_qubits=num_qubits,
        one_qubit=dict.fromkeys(range(num_qubits), 0.99),
        two_qubit=dict.fromkeys(heavy_hex_coupling, 0.99),
        measurement_confidences=dict.fromkeys(range(num_qubits), 0.99),
        basis_gates=["cz", "id", "rx", "rz", "rzz", "sx", "x"],
        t1=dict.fromkeys(range(num_qubits), 0.0001),
        t2=dict.fromkeys(range(num_qubits), 0.0002),
    )


def test_simple_qaoa(simple_generator: QuboGenerator) -> None:
    """Tests the construction of a simple QAOA circuit without qubit reuse and with barriers.

    Args:
        simple_generator (QuboGenerator): A simple QuboGenerator fixture.
    """
    circuit = simple_generator.construct_qaoa_circuit(do_reuse=False, include_barriers=True)
    expected_qubits = 8
    assert circuit.num_qubits == expected_qubits
    ops = circuit.count_ops()
    assert ops["barrier"] == 3
    assert ops["h"] == expected_qubits
    assert ops["rx"] == expected_qubits
    assert ops["rzz"] == 10


def test_simple_qaoa_no_barriers(simple_generator: QuboGenerator) -> None:
    """Tests the construction of a simple QAOA circuit without qubit reuse and without barriers.

    Args:
        simple_generator (QuboGenerator): A simple QuboGenerator fixture.
    """
    circuit = simple_generator.construct_qaoa_circuit(do_reuse=False, include_barriers=False)
    expected_qubits = 8
    assert circuit.num_qubits == expected_qubits
    ops = circuit.count_ops()
    assert "barrier" not in ops
    assert ops["h"] == expected_qubits
    assert ops["rx"] == expected_qubits
    assert ops["rzz"] == 10


def test_qaoa_with_reuse(simple_generator: QuboGenerator) -> None:
    """Tests the construction of a QAOA circuit with qubit reuse and without barriers.

    Args:
        simple_generator (QuboGenerator): A simple QuboGenerator fixture.
    """
    circuit = simple_generator.construct_qaoa_circuit(do_reuse=True, include_barriers=False)
    expected_qubits = 3
    expected_variables = 8
    assert circuit.num_qubits == expected_qubits
    ops = circuit.count_ops()
    assert "barrier" not in ops
    assert ops["h"] == expected_variables
    assert ops["rx"] == expected_variables
    assert ops["reset"] == expected_variables - expected_qubits
    assert ops["rzz"] == 10


def test_heavy_hex_qaoa(simple_generator: QuboGenerator, simple_device: Calibration) -> None:
    """Tests the construction of a heavy-hex optimized QAOA circuit.

    Args:
        simple_generator (QuboGenerator): A simple QuboGenerator fixture.
        simple_device (Calibration): A simple device (heavy-hex) calibration fixture.
    """
    circuit = simple_generator.construct_embedded_qaoa_circuit(simple_device)
    assert circuit.num_qubits == simple_device.num_qubits
    ops = circuit.count_ops()
    expected_variables = 8
    assert ops["h"] == expected_variables
    assert ops["rx"] == expected_variables
    assert ops["rzz"] == 10


@pytest.fixture
def weighted_generator() -> QuboGenerator:
    """Provides a QuboGenerator instance with distinct linear and quadratic coefficients.

    Returns:
        QuboGenerator: An instance of QuboGenerator with the objective function 3 x1 x2 - 2 x2 x3 + 5 x1 - 4 x3.
    """
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    return QuboGenerator(3 * x1 * x2 - 2 * x2 * x3 + 5 * x1 - 4 * x3)


def _qubo_costs(qubo: npt.NDArray[np.int_ | np.float64]) -> npt.NDArray[np.float64]:
    """Computes the QUBO cost for each computational basis state in Qiskit's little-endian order.

    Args:
        qubo (npt.NDArray[np.int_ | np.float64]): The upper-triangular QUBO matrix.

    Returns:
        npt.NDArray[np.float64]: The cost of each basis state.
    """
    n = len(qubo)
    bits = np.array([[(k >> i) & 1 for i in range(n)] for k in range(2**n)])
    return np.array([b @ qubo @ b for b in bits])


def _cost_terms(circuit: qiskit.QuantumCircuit, gamma: float) -> dict[frozenset[int], float]:
    """Collects the `rz` and `rzz` angles of a QAOA circuit by the QUBO variables that the gates act on.

    A physical qubit holds the variable that the next measurement on that qubit writes to.

    Args:
        circuit (qiskit.QuantumCircuit): The QAOA circuit.
        gamma (float): The value to assign to the `gamma` parameter.

    Returns:
        dict[frozenset[int], float]: The total angle for each set of variable indices.
    """
    gamma_parameter = next(p for p in circuit.parameters if p.name == "gamma")
    data = circuit.data
    terms: dict[frozenset[int], float] = {}
    for position, instruction in enumerate(data):
        if instruction.operation.name not in {"rz", "rzz"}:
            continue
        variables = set()
        for qubit in instruction.qubits:
            measurement = next(
                later for later in data[position + 1 :] if later.operation.name == "measure" and qubit in later.qubits
            )
            variables.add(circuit.find_bit(measurement.clbits[0]).index)
        angle = float(instruction.operation.params[0].bind({gamma_parameter: gamma}))
        key = frozenset(variables)
        terms[key] = terms.get(key, 0.0) + angle
    return terms


def test_qaoa_cost_layer_encodes_qubo(weighted_generator: QuboGenerator) -> None:
    """Tests that the QAOA cost layer applies the phase exp(-i c gamma f(x)) for one fixed scale c.

    Args:
        weighted_generator (QuboGenerator): A QuboGenerator fixture with distinct coefficients.
    """
    circuit = weighted_generator.construct_qaoa_circuit(do_reuse=False)
    gamma_parameter = next(p for p in circuit.parameters if p.name == "gamma")
    gamma = 0.05
    cost_layer = qiskit.QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        if instruction.operation.name in {"rz", "rzz"}:
            cost_layer.append(instruction.operation, instruction.qubits)
    phases = np.diag(Operator(cost_layer.assign_parameters({gamma_parameter: gamma})).data)

    cost_differences = _qubo_costs(weighted_generator.construct_qubo_matrix())
    cost_differences -= cost_differences[0]
    phase_differences = np.angle(phases / phases[0])
    reference = int(np.argmax(np.abs(cost_differences)))
    scale = -phase_differences[reference] / (gamma * cost_differences[reference])

    assert scale > 0
    np.testing.assert_allclose(phases / phases[0], np.exp(-1j * scale * gamma * cost_differences), atol=1e-9)


def test_qaoa_with_reuse_matches_cost_layer(weighted_generator: QuboGenerator) -> None:
    """Tests that qubit reuse applies the same cost terms to each QUBO variable as the circuit without reuse.

    Args:
        weighted_generator (QuboGenerator): A QuboGenerator fixture with distinct coefficients.
    """
    gamma = 0.7
    without_reuse = _cost_terms(weighted_generator.construct_qaoa_circuit(do_reuse=False), gamma)
    with_reuse = _cost_terms(weighted_generator.construct_qaoa_circuit(do_reuse=True), gamma)
    assert with_reuse.keys() == without_reuse.keys()
    for variables, angle in without_reuse.items():
        assert with_reuse[variables] == pytest.approx(angle)


def test_qaoa_includes_variables_without_interactions() -> None:
    """Tests that the QAOA circuit has a qubit for a variable that only has a linear term."""
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    generator = QuboGenerator(x1 * x2 + 2 * x3)
    circuit = generator.construct_qaoa_circuit(do_reuse=False)
    assert circuit.num_qubits == 3
    assert frozenset({2}) in _cost_terms(circuit, 1.0)


@pytest.mark.parametrize("do_reuse", [False, True])
def test_qaoa_mixer_layer(weighted_generator: QuboGenerator, do_reuse: bool) -> None:
    """Tests that each mixer gate applies exp(-i beta X) to its qubit.

    Args:
        weighted_generator (QuboGenerator): A QuboGenerator fixture with distinct coefficients.
        do_reuse (bool): Whether the circuit reuses qubits.
    """
    circuit = weighted_generator.construct_qaoa_circuit(do_reuse=do_reuse)
    beta_parameter = next(p for p in circuit.parameters if p.name == "beta")
    beta = 0.4
    expected = np.cos(beta) * np.eye(2) - 1j * np.sin(beta) * np.array([[0, 1], [1, 0]])
    mixer_gates = [instruction.operation for instruction in circuit.data if instruction.operation.name == "rx"]
    assert len(mixer_gates) == len(weighted_generator.construct_qubo_matrix())
    for gate in mixer_gates:
        bound = qiskit.QuantumCircuit(1)
        bound.append(gate, [0])
        np.testing.assert_allclose(Operator(bound.assign_parameters({beta_parameter: beta})).data, expected, atol=1e-9)
