from mqt.problemsolver.csp import CSP


def test_csp():
    csp = CSP()
    sum_s0 = 1
    sum_s1 = 3
    sum_s2 = 3
    sum_s3 = 1
    list_of_constraints = csp.get_kakuro_constraints(
        sum_s0=sum_s0, sum_s1=sum_s1, sum_s2=sum_s2, sum_s3=sum_s3
    )
    res = csp.solve(constraints=list_of_constraints, quantum_algorithm="Grover")
    assert res == (0, 3, 1, 0)

    sum_s0 = 3
    sum_s1 = 5
    sum_s2 = 3
    sum_s3 = 5
    list_of_constraints = csp.get_kakuro_constraints(
        sum_s0=sum_s0, sum_s1=sum_s1, sum_s2=sum_s2, sum_s3=sum_s3
    )
    res = csp.solve(constraints=list_of_constraints, quantum_algorithm="Grover")
    assert res == (0, 3, 3, 2)
