from src.mqt.problemsolver.kakuro import Kakuro


def test_kakuro():
    kakuro = Kakuro()
    assert kakuro.solve(1, 3, 3, 1, quantum_algorithm="Grover") == (0, 3, 1, 0)
    assert kakuro.solve(3, 5, 3, 5, quantum_algorithm="Grover") == (0, 3, 3, 2)
