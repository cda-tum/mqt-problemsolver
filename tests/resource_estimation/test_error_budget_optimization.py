from __future__ import annotations

from mqt.problemsolver.resource_estimation.error_budget_optimization import evaluate, generate_data, train


def test_error_budget_optimization() -> None:
    total_error_budget = 0.1
    benchmarks_and_sizes = [("dj", [3, 4, 5, 6, 7, 8, 9, 10])]
    data = generate_data(
        total_error_budget=total_error_budget,
        number_of_randomly_generated_distributions=10,
        benchmarks_and_sizes=benchmarks_and_sizes,
    )
    model, x_test, y_test = train(data)
    y_pred = model.predict(x_test)
    evaluate(x_test, y_pred, total_error_budget)
    evaluate(x_test, y_test, total_error_budget)
