from mqt.problemsolver.tsp import TSP


def test_kakuro() -> None:
    tsp = TSP()
    res = tsp.solve(1, 2, 3, 4, 5, 6, quantum_algorithm="QPE", objective_function="shortest_path")
    assert res is not None
    assert res == [3, 1, 2, 4]
    res = tsp.solve(6, 9, 2, 1, 8, 4, quantum_algorithm="QPE", objective_function="shortest_path")
    assert res is not None
    assert res == [4, 1, 2, 3]
