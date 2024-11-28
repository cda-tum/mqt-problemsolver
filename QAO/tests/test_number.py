"""File for making some tests during the Development"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import Any

# for managing symbols
from mqt.qao.karp import KarpNumber
from mqt.qao.problem import Problem


def test_sat_initialization():
    """Test the initialization of the SAT problem."""
    input_data: list[Any] = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    problem = KarpNumber.sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for SAT initialization"


def test_print_solution():
    """Unit test for the print_solution method."""

    captured_output = StringIO()
    sys.stdout = captured_output

    problem_name = "Test Problem"
    file_name = "test_file"
    solution = "This is the solution."
    summary = "Summary details."

    expected_output = (
        "Test Problemtest_file\n=====================\nThis is the solution.\n---------------------\nSummary details.\n"
    )

    KarpNumber.print_solution(problem_name=problem_name, file_name=file_name, solution=solution, summary=summary)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output == expected_output, "The printed output does not match the expected result."


def test_convert_dict_to_string():
    """Test for the convert_dict_to_string method."""

    valid_solution_dict = {"Valid Solution": True, "Details": {"Constraint 1": "Passed", "Constraint 2": "Passed"}}
    expected_valid_output = "Valid Solution\nDetails: {'Constraint 1': 'Passed', 'Constraint 2': 'Passed'}"
    result_valid = KarpNumber.convert_dict_to_string(valid_solution_dict)

    differences = []
    for i, (char1, char2) in enumerate(zip(result_valid, expected_valid_output)):
        if char1 != char2:
            differences.append((i, char1, char2))

    assert result_valid == expected_valid_output

    invalid_solution_dict = {"Valid Solution": False, "Errors": {"Constraint 1": "Failed", "Constraint 2": ""}}
    expected_invalid_output = "Invalid Solution\nErrors: {'Constraint 1': 'Failed', 'Constraint 2': ''}"
    result_invalid = KarpNumber.convert_dict_to_string(invalid_solution_dict)

    assert result_invalid == expected_invalid_output, "Output for invalid solution does not match the expected result."


def test_save_solution():
    """Test for the save_solution method."""

    # Test data
    problem_name = "Test Problem"
    file_name = "test_file"
    solution = "This is the solution."
    summary = "Summary details."
    txt_outputname = "test_output.txt"

    # Expected content
    expected_content = (
        "Test Problemtest_file\n=====================\nThis is the solution.\n---------------------\nSummary details."
    )

    # Call the method
    KarpNumber.save_solution(
        problem_name=problem_name,
        file_name=file_name,
        solution=solution,
        summary=summary,
        txt_outputname=txt_outputname,
    )

    # Verify file content
    with Path(txt_outputname).open(encoding="utf-8") as f:
        content = f.read()
        assert content == expected_content, "File content does not match the expected result."

    # Clean up by removing the created file
    Path(txt_outputname).unlink()


def test_check_integer_programming():
    """Test for the check_integer_programming method."""

    s_valid = [[1, 2], [3, 4]]
    b_valid = [5, 11]
    x_valid = [1.0, 2.0]  # 1*1 + 2*2 = 5 and 3*1 + 4*2 = 11
    result_valid = KarpNumber.check_integer_programming(s_valid, b_valid, x_valid)
    assert result_valid == {"Valid Solution": True}, "Failed valid solution test"

    s_invalid = [[1, 2], [3, 4]]
    b_invalid = [5, 11]
    x_invalid = [0.0, 1.0]  # 1*0 + 2*1 != 5
    result_invalid = KarpNumber.check_integer_programming(s_invalid, b_invalid, x_invalid)
    assert not result_invalid["Valid Solution"]


def test_sat_solving_basic():
    """Test the basic solving of the SAT problem without specifying solver parameters."""
    input_data: list[Any] = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = KarpNumber.sat(input_data, solve=True)
    assert isinstance(solution, dict), "Expected a dictionary as the solution"
    assert all(isinstance(value, float) for key, value in solution.items()), (
        "Expected solution to contain variable names as keys and floats as values"
    )


def test_three_sat_initialization():
    """Test the initialization of the 3-SAT problem."""
    input_data: list[Any] = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    problem = KarpNumber.three_sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for 3-SAT initialization"


def test_three_sat_solving_basic():
    """Test the basic solving of the 3-SAT problem without specifying solver parameters."""
    input_data: list[Any] = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = KarpNumber.three_sat(input_data, solve=True)

    assert isinstance(solution, dict), "Expected a dictionary as the solution"
    assert all(isinstance(value, float) for key, value in solution.items()), (
        "Expected solution to contain variable names as keys and floats as values"
    )


def test_sat_solution_validation_correct():
    """Test validation for a correct SAT solution."""
    input_data: list[Any] = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_sat_solution_validation_incorrect():
    """Test validation for an incorrect SAT solution."""
    input_data: list[Any] = [["a", "!b"], ["b", "c"], ["!a", "d"]]
    solution = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to unsatisfied clauses"


def test_three_sat_solution_validation_correct():
    """Test validation for a correct 3-SAT solution."""
    input_data: list[Any] = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0, "e": 1.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_three_sat_solution_validation_incorrect():
    """Test validation for an incorrect 3-SAT solution."""
    input_data: list[Any] = [["a", "!b", "c"], ["b", "c", "d"], ["!a", "!d", "e"]]
    solution = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.0}
    validation = KarpNumber.check_three_sat_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to unsatisfied clauses"


def test_sat_empty_input():
    """Test handling of an empty input for the SAT problem."""
    input_data: list[Any] = []
    problem = KarpNumber.sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_three_sat_empty_input():
    """Test handling of an empty input for the 3-SAT problem."""
    input_data: list[Any] = []
    problem = KarpNumber.three_sat(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_number_partition_initialization():
    """Test the initialization of the number partition problem."""
    input_data: list[Any] = [3, 1, 4, 2, 2]
    problem = KarpNumber.number_partition(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for number partition initialization"


def test_number_partition_solving_basic():
    """Test the basic solving of the number partition problem."""
    input_data: list[Any] = [3, 1, 4, 2, 2]
    solution = KarpNumber.number_partition(input_data, solve=True)

    if not isinstance(solution, tuple):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    set_1, set_2 = solution

    assert isinstance(set_1, list)
    assert isinstance(set_2, list)


def test_number_partition_balanced_solution():
    """Test that the partitioned sets are approximately balanced."""
    input_data: list[Any] = [3, 1, 4, 2, 2]
    solution = KarpNumber.number_partition(input_data, solve=True)

    if not isinstance(solution, tuple):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    set_1, set_2 = solution

    sum_set_1 = sum(set_1)
    sum_set_2 = sum(set_2)
    assert abs(sum_set_1 - sum_set_2) <= 1, "Expected partitions to have approximately balanced sums"


def test_number_partition_solution_validation_correct():
    """Test validation for a correct number partition solution."""
    input_data: list[Any] = [3, 1, 4, 2, 2]
    solution = {"s_1": 1.0, "s_2": 1.0, "s_3": -1.0, "s_4": -1.0, "s_5": 1.0}  # thiswould create two balanced sets
    validation = KarpNumber.check_number_partition_solution(input_data, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_number_partition_solution_validation_incorrect():
    """Test validation for an incorrect number partition solution."""
    input_data: list[Any] = [3, 1, 4, 2, 2]
    solution = {"s_1": 1.0, "s_2": 1.0, "s_3": 1.0, "s_4": -1.0, "s_5": -1.0}  # imbalanced partition
    validation = KarpNumber.check_number_partition_solution(input_data, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to imbalance in sums"


def test_number_partition_empty_input():
    """Test handling of an empty input for the number partition problem."""
    input_data: list[Any] = []
    problem = KarpNumber.number_partition(input_data, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance even with empty input"


def test_job_sequencing_initialization():
    """Test the initialization of the job sequencing problem."""
    job_lengths: list[Any] = [3, 1, 2, 2]
    m: int = 2
    problem = KarpNumber.job_sequencing(job_lengths, m, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for job sequencing initialization"


def test_job_sequencing_solving_basic():
    """Test the basic solving of the job sequencing problem."""
    job_lengths: list[Any] = [3, 1, 2, 2]
    m: int = 2
    solution = KarpNumber.job_sequencing(job_lengths, m, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(isinstance(machine_jobs, list) for machine_jobs in solution), "Each machine's job list should be a list"
    assert len(solution) == m, f"Expected {m} machines in the solution"


def test_job_sequencing_with_single_machine():
    """Test job sequencing when only one machine is available."""
    job_lengths: list[Any] = [3, 5, 2, 7, 1]
    m: int = 1
    solution = KarpNumber.job_sequencing(job_lengths, m, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert sum(job_lengths) == sum(job_lengths[job] for job in solution[0]), (
        "All jobs should be assigned to the single machine available"
    )


def test_knapsack_initialization():
    """Test the initialization of the knapsack problem."""
    items: list[Any] = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight: int = 5
    problem = KarpNumber.knapsack(items, max_weight, solve=False)
    assert isinstance(problem, Problem), "Expected a Problem instance for knapsack initialization"


def test_knapsack_solving_basic():
    """Test the basic solving of the knapsack problem."""
    items: list[Any] = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight: int = 5
    solution = KarpNumber.knapsack(items, max_weight, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)
    assert isinstance(solution, list), "Expected a list as the solution"
    assert all(len(item) == 2 for item in solution), "Each solution item should be a tuple of (weight, value)"


def test_knapsack_solution_optimization():
    """Test that the knapsack solution maximizes the value without exceeding max weight."""
    items = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight = 5
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    total_weight = sum(item[0] for item in solution)
    total_value = sum(item[1] for item in solution)

    assert total_weight <= max_weight, "Total weight should not exceed the maximum allowed weight"
    assert total_value == 8, "Expected maximum value to be 7 for this input"


def test_knapsack_solution_validation_correct():
    """Test validation for a correct knapsack solution."""
    max_weight: int = 5
    solution: list[Any] = [(2, 3), (3, 4)]
    validation = KarpNumber.check_knapsack_solution(max_weight, solution)
    assert validation["Valid Solution"], "Expected solution to be valid"


def test_knapsack_solution_validation_incorrect():
    """Test validation for an incorrect knapsack solution (overweight)."""
    max_weight: int = 5
    solution: list[Any] = [(3, 4), (4, 5)]
    validation = KarpNumber.check_knapsack_solution(max_weight, solution)
    assert not validation["Valid Solution"], "Expected solution to be invalid due to exceeding weight limit"


def test_knapsack_with_zero_max_weight():
    """Test knapsack when max weight is zero, expecting an empty solution."""
    items: list[Any] = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight: int = 0
    solution = KarpNumber.knapsack(items, max_weight, solve=True)
    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    assert solution == [], "Expected an empty solution when max weight is zero"


def test_knapsack_with_large_max_weight():
    """Test knapsack with a large max weight, expecting all items to be included if possible."""
    items: list[Any] = [(3, 4), (2, 3), (4, 5), (5, 8)]
    max_weight: int = 15
    solution = KarpNumber.knapsack(items, max_weight, solve=True)

    if not isinstance(solution, list):
        msg = f"Expected solution to be a list, but got {type(solution).__name__}"
        raise TypeError(msg)

    total_weight: int = sum(item[0] for item in solution)
    total_value: int = sum(item[1] for item in solution)

    assert total_weight <= max_weight, "Total weight should not exceed the maximum allowed weight"
    assert total_value <= 20, "Expected total value to be the sum of all item values"
