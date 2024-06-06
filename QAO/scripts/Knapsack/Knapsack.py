from __future__ import annotations

import locale
import os
from pathlib import Path
from typing import cast

import numpy as np
from sympy import Expr

# for managing symbols
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables

lambdas_method = [
    "upper_bound_only_positive",
    "maximum_coefficient",
    "VLM",
    "MOMC",
    "MOC",
    "upper lower bound naive",
    "upper lower bound posiform and negaform method",
]
lambdas_conf = [
    ("upper_bound_only_positive", "sequential penalty increase"),
    ("maximum_coefficient", "sequential penalty increase"),
    ("VLM", "sequential penalty increase"),
    ("MOMC", "sequential penalty increase"),
    ("MOC", "sequential penalty increase"),
    ("upper lower bound naive", "sequential penalty increase"),
    ("upper lower bound posiform and negaform method", "sequential penalty increase"),
    ("upper_bound_only_positive", "scaled sequential penalty increase"),
    ("maximum_coefficient", "scaled sequential penalty increase"),
    ("VLM", "scaled sequential penalty increase"),
    ("MOMC", "scaled sequential penalty increase"),
    ("MOC", "scaled sequential penalty increase"),
    ("upper lower bound naive", "scaled sequential penalty increase"),
    ("upper lower bound posiform and negaform method", "scaled sequential penalty increase"),
    ("upper_bound_only_positive", "binary search penalty algorithm"),
    ("maximum_coefficient", "binary search penalty algorithm"),
    ("VLM", "binary search penalty algorithm"),
    ("MOMC", "binary search penalty algorithm"),
    ("MOC", "binary search penalty algorithm"),
    ("upper lower bound naive", "binary search penalty algorithm"),
    ("upper lower bound posiform and negaform method", "binary search penalty algorithm"),
]

files = os.listdir("Data/")
for file in files:
    print(file)
    with Path("./Data/" + file).open("r", encoding=locale.getpreferredencoding(False)) as f:
        lines = f.readlines()
        el = lines[0].split()
        objects = int(el[0])
        W_max = float(el[1])
        w = []
        p = []
        for i in range(1, len(lines)):
            el = lines[i].split()
            p.append(float(el[0]))
            w.append(float(el[1]))
        p_arr = np.asarray(p)
        w_arr = np.asarray(w)

    for lambdas in lambdas_method:
        variables = Variables()
        obj = variables.add_binary_variables_array("obj", [objects])
        objective_function = ObjectiveFunction()
        objective_function.add_objective_function(cast(Expr, np.dot(np.transpose(obj), p_arr)), minimization=False)
        constraint = Constraints()
        constraint.add_constraint(str(np.dot(np.transpose(obj), w_arr)) + " <= " + format(W_max))
        problem = Problem()
        problem.create_problem(variables, constraint, objective_function)
        solver = Solver()
        solution = solver.solve_simulated_annealing(
            problem, lambda_strategy=lambdas, auto_setting=True, save_time=True, max_lambda_update=1
        )
        if not isinstance(solution, bool):
            all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
            solution.valid_solutions()
            solution.wring_json_reports(
                filename="simulated_annealing_knapsack_" + file + "_" + lambdas, problem_features=True
            )

    for lambdas_m, lambdas_update in lambdas_conf:
        variables = Variables()
        obj = variables.add_binary_variables_array("obj", [objects])
        objective_function = ObjectiveFunction()
        objective_function.add_objective_function(cast(Expr, np.dot(np.transpose(obj), p_arr)), minimization=False)
        constraint = Constraints()
        constraint.add_constraint(str(np.dot(np.transpose(obj), w_arr)) + " <= " + format(W_max))
        problem = Problem()
        problem.create_problem(variables, constraint, objective_function)
        solver = Solver()
        solution = solver.solve_simulated_annealing(
            problem,
            lambda_strategy=lambdas_m,
            lambda_update_mechanism=lambdas_update,
            auto_setting=True,
            save_time=True,
            max_lambda_update=1,
        )
        if not isinstance(solution, bool):
            all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
            print(all_satisfy, each_satisfy)
            solution.valid_solutions()
            solution.wring_json_reports(
                filename="simulated_annealing_knapsack_" + file + "_" + lambdas_m + "_" + lambdas_update,
                problem_features=True,
            )
