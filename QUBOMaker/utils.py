import numpy as np

from typing import Iterable
from IPython.display import display, Math

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
    
def optimise_classically(qubo: np.mat) -> tuple[list[int], float]:
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
        
    all_tests = [from_binary(i, 16) for i in range(2**16)]

    best_test = None
    best_score = 999999999999

    import numpy as np
    for i, test in enumerate(all_tests):
        x = np.array(test)
        score = np.matmul(x.T, np.matmul(qubo, x))
        if best_score > score:
            best_score = score
            best_test = test
        if i % 25 == 0:
            print(f"{i / len(all_tests) * 100}%")
            
    return (best_test, best_score)