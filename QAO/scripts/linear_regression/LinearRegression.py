from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sympy import Expr

# for managing symbols
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables

df = pd.read_csv("iris_csv.csv")
df = shuffle(df)
d = len(df.columns) - 1
N = len(df[df.keys()[1]])
feat = 2
X = np.zeros([N, d])
Y = np.zeros(N)
TrainingPercentage = 0.7
N_Traning = int(N * TrainingPercentage)
N_Test = N - N_Traning
i = 0
for key in df:
    j = 0
    if key != "class":
        for el in df[key]:
            X[j, i] = el
            j += 1
        i += 1
    else:
        for el in df[key]:
            if el == "Iris-setosa":
                Y[j] = 1
            else:
                Y[j] = -1
            j += 1
pca = PCA(n_components=feat)
X = pca.fit_transform(X=X, y=Y)
X_traning = np.hstack((X[:N_Traning, :], np.ones((N_Traning, 1))))
X_test = np.hstack((X[N_Traning:, :], np.ones((N_Test, 1))))

Y_traning = Y[:N_Traning]
Y_test = Y[N_Traning:]
variables = Variables()
w = variables.add_continuous_variables_array("w", [feat + 1, 1], -0.25, 0.25, 0.25)
objective_function = ObjectiveFunction()
objective_function.add_objective_function(
    cast(
        Expr,
        (
            np.dot(np.dot(np.dot(np.transpose(w), np.transpose(X_traning)), X_traning), w)
            - 2 * np.dot(np.dot(np.transpose(w), np.transpose(X_traning)), Y_traning)
            + np.dot(np.transpose(Y_traning), Y_traning)
        )[0, 0],
    )
)
constraint = Constraints()
problem = Problem()
problem.create_problem(variables, constraint, objective_function)
solver = Solver()

PredictionRes = {}

solution = solver.solve_simulated_annealing(
    problem, auto_setting=True, save_time=True, max_lambda_update=0, save_compilation_time=True
)
if not isinstance(solution, bool):
    all_satisfy, each_satisfy = solution.check_constraint_optimal_solution()
    print(all_satisfy, each_satisfy)
    solution.valid_solutions()
    print(solution.best_energy)
    solution.wring_json_reports(filename="simulated_annealing_linear_regression_Iris", problem_features=True)
    w_conf = solution.best_solution["w"]
    Y_obtained_training = np.dot(X_traning, w_conf)
    TP_tr = 0
    TN_tr = 0
    FP_tr = 0
    FN_tr = 0
    for i in range(N_Traning):
        if Y_obtained_training[i] > 0:
            if Y_traning[i] == 1:
                TP_tr += 1
            else:
                FP_tr += 1
        elif Y_traning[i] == 1:
            FN_tr += 1
        else:
            TN_tr += 1

    try:
        Accuracy_tr = (TN_tr + TP_tr) / (TN_tr + TP_tr + FP_tr + FN_tr)
    except ZeroDivisionError:
        Accuracy_tr = 0

    try:
        Precision_tr = TP_tr / (TP_tr + FP_tr)
    except ZeroDivisionError:
        Precision_tr = 0

    try:
        Recall_tr = TP_tr / (TP_tr + FN_tr)
    except ZeroDivisionError:
        Recall_tr = 0

    try:
        F1_score_tr = 2 * Precision_tr * Recall_tr / (Precision_tr + Recall_tr)
    except ZeroDivisionError:
        F1_score_tr = 0

    PredictionRes["Accuracy training"] = Accuracy_tr
    PredictionRes["Precision training"] = Precision_tr
    PredictionRes["Recall training"] = Recall_tr
    PredictionRes["F1 training"] = F1_score_tr
    Y_obtained_test = np.dot(X_test, w_conf)
    print(Accuracy_tr, Precision_tr, Recall_tr)
    TP_t = 0
    TN_t = 0
    FP_t = 0
    FN_t = 0
    for i in range(N_Test - 1):
        if Y_obtained_test[i] > 0:
            if Y_test[i] == 1:
                TP_t += 1
            else:
                FP_t += 1
        elif Y_test[i] == 1:
            FN_t += 1
        else:
            TN_t += 1

    try:
        Accuracy_t = (TN_t + TP_t) / (TN_t + TP_t + FP_t + FN_t)
    except ZeroDivisionError:
        Accuracy_t = 0

    try:
        Precision_t = TP_t / (TP_t + FP_t)
    except ZeroDivisionError:
        Precision_t = 0

    try:
        Recall_t = TP_t / (TP_t + FN_t)
    except ZeroDivisionError:
        Recall_t = 0

    try:
        F1_score_t = 2 * Precision_t * Recall_t / (Precision_t + Recall_t)
    except ZeroDivisionError:
        F1_score_t = 0
    print(Accuracy_t, Precision_t, Recall_t)
    PredictionRes["Accuracy test"] = Accuracy_t
    PredictionRes["Precision test"] = Precision_t
    PredictionRes["Recall test"] = Recall_t
    PredictionRes["F1 test"] = F1_score_t


df1 = pd.DataFrame.from_dict(PredictionRes, orient="index")
df1.to_csv("PredictionResSimulatedAnnealing.csv")
