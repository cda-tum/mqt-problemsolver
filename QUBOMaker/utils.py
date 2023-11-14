from typing import Iterable
import numpy as np
from IPython.display import display, Math, clear_output
from ipywidgets import widgets

def print_matrix(array: Iterable[Iterable[float]]):
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'Q = \begin{bmatrix}'+matrix+r'\end{bmatrix}'))

def optimise_classically(qubo: np.ndarray,
                         show_progress_bar: bool = False) -> tuple[list[int], float]:
    progress_bar: widgets.FloatProgress | None = None
    if show_progress_bar:
        progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=1,
            description='Calculating:',
            bar_style='info',
            style={'bar_color': "#0055bb"},
            orientation='horizontal'
        )

    def from_binary(num: int, digits: int) -> list[int]:
        binary = []
        c = 0
        while num > 0:
            binary.append(num % 2)
            c += 1
            num = num // 2
        for _ in range(c, digits):
            binary.append(0)
        return binary

    all_tests = [from_binary(i, qubo.shape[0]) for i in range(2**qubo.shape[0])]

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
