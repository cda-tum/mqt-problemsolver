import functools
from typing import Any, Callable
import numpy as np

import arithmetic
from arithmetic import ArithmeticItem, Constant, Division, Exponentiation, LatexTransformer, Multiplication, SimplifyingTransformer, BreakDownSumsTransformer, ConstantCalculatingTransformer, AdditionLiftingTransformer, Variable, VisitingTransformer


class QUBOGenerator:
    objective_function: ArithmeticItem
    
    penalties: list[tuple[ArithmeticItem, int | None]] 

    def __init__(self, objective_function: ArithmeticItem) -> None:
        self.objective_function = objective_function
        self.penalties = []

    def add_penalty(self, penalty_function: ArithmeticItem, lam: int | None = None) -> None:
        self.penalties.append((penalty_function, lam))
        
    def construct(self) -> ArithmeticItem:
        return functools.reduce(lambda current, new: current + new[1] * new[0],
                        self.__select_lambdas(),
                        self.objective_function)
        
    def construct_expansion(self) -> ArithmeticItem:
        expression = self.construct()
        return SimplifyingTransformer().transform(
            ConstantCalculatingTransformer().transform(
                AdditionLiftingTransformer().transform(
                    BreakDownSumsTransformer().transform(expression)
                )
            )
        )
        
    def construct_qubo_matrix(self) -> np.mat:
        exp = self.construct_expansion()
        result = np.zeros((self.get_qubit_count(), self.get_qubit_count()))
        current = exp
        while isinstance(current, arithmetic.Addition):
            (factor, i, j) = self.__get_factor(current.left)
            if i == -1:
                i = j
            if j == -1:
                j = i
            if i == -1:
                current = current.right
                continue #constant
            if i <= j:            
                result[i - 1, j - 1] += factor
            else:
                result[j - 1, i - 1] += factor
            current = current.right
        (factor, i, j) = self.__get_factor(current)
        if i <= j:
            result[i - 1, j - 1] += factor
        else:
            result[j - 1, i - 1] += factor
        
        return result
    
    def __get_factor(self, expression: ArithmeticItem) -> tuple[int, int, int]:
        factor = 1
        i = -1
        j = -1
        
        if isinstance(expression, Variable):
            return (1, self.get_variable_index(expression), -1)
        
        if isinstance(expression, Constant):
            return (expression.value, -1, -1)
        
        if isinstance(expression, Exponentiation):
            if expression.exponent != Constant(2):
                raise ValueError()
            (factor, i, j) = self.__get_factor(expression.base)
            factor *= factor
            
        if isinstance(expression, Division):
            if not isinstance(expression.right, Constant):
                raise ValueError()
            (factor, i, j) = self.__get_factor(expression.left)
            factor /= expression.right
        
        if isinstance(expression, Multiplication):
            (f1, i1, j1) = self.__get_factor(expression.left)
            (f2, i2, j2) = self.__get_factor(expression.right)
            factor *= f1 * f2
            indices = [x for x in [i1, i2, j1, j2] if x != -1]
            for index in indices:
                if index == -1:
                    continue
                if i == -1:
                    i = index
                elif j == -1:
                    j = index
                else:
                    raise ValueError()
        
        return (factor, i, j)
        
    def __select_lambdas(self) -> list[tuple[ArithmeticItem, int]]:
        return self.penalties #TODO
    
    def get_qubit_count(self) -> int:
        return 0 #TODO
    
    def get_variable_index(self, variable: Variable) -> int:
        return 1
    
    def decode_bit_array(self, array: list[int]) -> Any:
        return ""