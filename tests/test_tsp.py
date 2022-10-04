from src.mqt.problemsolver.tsp import TSP


def test_kakuro():
    tsp = TSP()
    assert tsp.solve(1, 2, 3, 4, 5, 6, quantum_algorithm="QPE") == [3, 1, 2, 4]
    assert tsp.solve(6, 9, 2, 1, 8, 4, quantum_algorithm="QPE") == [4, 1, 2, 3]
