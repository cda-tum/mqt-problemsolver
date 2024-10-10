"""Provides utility functions that can be used with QuboMaker."""

from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

import numpy as np
import numpy.typing as npt
from IPython.display import Math, clear_output, display
from ipywidgets import widgets

if TYPE_CHECKING:
    from collections.abc import Iterable


@no_type_check
def print_matrix(array: Iterable[Iterable[float]]) -> None:
    """Print a matrix data structure in LaTeX format.

    Args:
        array (Iterable[Iterable[float]]): The matrix to be printed.
    """
    matrix = ""
    for row in array:
        try:
            for number in row:
                matrix += f"{number}&"
        except TypeError:
            matrix += f"{row}&"
        matrix = matrix[:-1] + r"\\"
    display(Math(r"Q = \begin{bmatrix}" + matrix + r"\end{bmatrix}"))


@no_type_check
def optimize_classically(
    qubo: npt.NDArray[np.int_ | np.float64], show_progress_bar: bool = False
) -> tuple[list[int], float]:
    """Classically optimizes a given QUBO problem of the form x^TQx.

    Args:
        qubo (npt.NDArray[np.int_  |  np.float64]): A matrix representing the QUBO problem.
        show_progress_bar (bool, optional): If True, shows a progress bar in jupyter notebooks during calculation. Defaults to False.

    Returns:
        tuple[list[int], float]: The optimal solution and its corresponding score.
    """
    progress_bar: widgets.FloatProgress | None = None
    if show_progress_bar:
        progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=1,
            description="Calculating:",
            bar_style="info",
            style={"bar_color": "#0055bb"},
            orientation="horizontal",
        )

    def int_to_fixed_length_binary(number: int, length: int) -> list[int]:
        binary_string = f"{number:b}"
        padding_zeros = max(0, length - len(binary_string))
        binary_string = "0" * padding_zeros + binary_string
        return [int(bit) for bit in binary_string]

    all_tests = [int_to_fixed_length_binary(i, qubo.shape[0]) for i in range(2 ** qubo.shape[0])]

    best_test: list[int] = []
    best_score = 999999999999

    for i, test in enumerate(all_tests):
        x = np.array(test)
        score = np.matmul(x.T, np.matmul(qubo, x))
        if best_score > score:
            best_score = score
            best_test = test
        if i % 2000 == 0 and show_progress_bar and progress_bar is not None:
            progress_bar.value = i / len(all_tests)
            clear_output(True)
            display(progress_bar)

    if show_progress_bar and progress_bar is not None:
        progress_bar.value = 1
        clear_output(True)
        display(progress_bar)
    return (best_test, best_score)
