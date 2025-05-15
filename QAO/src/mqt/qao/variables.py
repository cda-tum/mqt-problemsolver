# Import of the needed libraries
# pyqubo libraries for some steps of the problem formulation
# numpy for matrix management
from __future__ import annotations

from math import ceil, floor, log, log2, sqrt
from typing import Any

import numpy as np
from qubovert import boolean_var

# for managing symbols
from sympy import Symbol, expand, symbols


class Variables:
    """class for declaring all the variables,
    which are part of the problem.
    Declaration of heterogeneous type of
    variables is admitted and managed"""

    def __init__(self) -> None:
        """declaration of the variables dictionary
        expected format {var_name: variable_obj,
        matrix_var_name: matrix(variable_obj)}
        of the binary variables"""
        self.variables_dict: dict[str, Any] = {}
        self.binary_variables_name_weight: dict[str, Any] = {}

    def add_binary_variable(self, name: str) -> Symbol | bool:
        """function for adding a unipolar variable in the variable list

        Keyword arguments:
        name -- string containing the variable name

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: Symbol | bool
        var = Binary()
        symb = var.create(name)
        if name not in self.variables_dict:
            self.variables_dict[name] = var
            ret = symb
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_binary_variables_array(self, name: str, shape: list[int]) -> np.ndarray[Any, np.dtype[Any]] | bool:
        """function for adding a unipolar variable array in the variable list

        Keyword arguments:
        name -- string containing the variable name
        shape -- tuple containing the size of the array (for the current moment 3D vectors are the maximum supported)

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: np.ndarray[Any, np.dtype[Any]] | bool
        arr: list[Any] = []
        symb_arr: list[Any] = []
        if len(shape) == 1:
            for i in range(shape[0]):
                arr.append(Binary())
                symb_arr.append(arr[i].create(name + "_" + format(i)))
        elif len(shape) == 2:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append(Binary())
                    symb_arr[i].append(arr[i][j].create(name + "_" + format(i) + "_" + format(j)))
        elif len(shape) == 3:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append([])
                    symb_arr[i].append([])
                    for k in range(shape[2]):
                        arr[i][j].append(Binary())
                        symb_arr[i][j].append(
                            arr[i][j][k].create(name + "_" + format(i) + "_" + format(j) + "_" + format(k))
                        )
        else:
            print("Arrays of dimension higher than 3 are currently not supported\n")
            return False

        if name not in self.variables_dict:
            self.variables_dict[name] = arr
            ret = np.array(symb_arr)

        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_spin_variable(self, name: str) -> Symbol | bool:
        """function for adding a bipolar binary variable in the variable list

        Keyword arguments:
        name -- string containing the variable name

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: Symbol | bool
        var = Binary()
        symb = var.create(name, False)
        if name not in self.variables_dict:
            self.variables_dict[name] = var
            ret = symb
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_spin_variables_array(self, name: str, shape: list[int]) -> np.ndarray[Any, np.dtype[Any]] | bool:
        """function for adding a bipolar variable array in the variable list

        Keyword arguments:
        name -- string containing the variable name
        shape -- tuple containing the size of the array (for the current moment 3D vectors are the maximum supported)


        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: np.ndarray[Any, np.dtype[Any]] | bool
        arr: list[Any] = []
        symb_arr: list[Any] = []
        if len(shape) == 1:
            for i in range(shape[0]):
                arr.append(Binary())
                symb_arr.append(arr[i].create(name + "_" + format(i), False))
        elif len(shape) == 2:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append(Binary())
                    symb_arr[i].append(arr[i][j].create(name + "_" + format(i) + "_" + format(j), False))
        elif len(shape) == 3:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append([])
                    symb_arr[i].append([])
                    for k in range(shape[2]):
                        arr[i][j].append(Binary())
                        symb_arr[i][j].append(
                            arr[i][j][k].create(
                                name + "_" + format(i) + "_" + format(j) + "_" + format(k),
                                False,
                            )
                        )
        else:
            print("Arrays of dimension higher than 3 are currently not supported\n")
            return False

        if name not in self.variables_dict:
            self.variables_dict[name] = arr
            ret = np.array(symb_arr)
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_discrete_variable(self, name: str, values: list[float]) -> Symbol | bool:
        """function for adding a discrete variable in the variable list

        Keyword arguments:
        name -- string containing the variable name
        values -- list of float containing the values that the variable can assume

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: Symbol | bool
        var = Discrete()
        symb = var.create(name, values)
        if name not in self.variables_dict:
            self.variables_dict[name] = var
            ret = symb
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_discrete_variables_array(
        self, name: str, shape: list[int], values: list[float]
    ) -> np.ndarray[Any, np.dtype[Any]] | bool:
        """function for adding a discrete variable array in the variable list

        Keyword arguments:
        name -- string containing the variable name
        shape -- tuple containing the size of the array (for the current moment 3D vectors are the maximum supported)
        values -- list of float containing the values that the variables can assume

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: np.ndarray[Any, np.dtype[Any]] | bool
        arr: list[Any] = []
        symb_arr: list[Any] = []
        if len(shape) == 1:
            for i in range(shape[0]):
                arr.append(Discrete())
                symb_arr.append(arr[i].create(name + "_" + format(i), values))
        elif len(shape) == 2:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append(Discrete())
                    symb_arr[i].append(arr[i][j].create(name + "_" + format(i) + "_" + format(j), values))
        elif len(shape) == 3:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append([])
                    symb_arr[i].append([])
                    for k in range(shape[2]):
                        arr[i][j].append(Discrete())
                        symb_arr[i][j].append(
                            arr[i][j][k].create(
                                name + "_" + format(i) + "_" + format(j) + "_" + format(k),
                                values,
                            )
                        )
        else:
            print("Arrays of dimension higher than 3 are currently not supported\n")
            ret = False

        if name not in self.variables_dict:
            self.variables_dict[name] = arr
            ret = np.array(symb_arr)
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_continuous_variable(
        self,
        name: str,
        min_val: float,
        max_val: float,
        precision: float,
        distribution: str = "uniform",
        encoding_mechanism: str = "",
    ) -> Symbol | bool:
        """function for adding a continuous variable in the variable list

        Keyword arguments:
        name -- string containing the variable name
        min_val -- float indicating the lower bound of the variable range
        max_val -- float indicating the upper bound of the variable range
        precision -- integer indicating the number of elements for sampling is a non-uniform distribution
                     is assumed and a float indicating the distance among two representable values if a
                     uniform distribution is assumed
        distribution -- string indicating the expected distribution of the value in the range
        encoding_mechanism -- string for forcing a specific encoding mechanism (prevelently for debugging purpose)
                                if empty the best encoding mechanism is decided by the toolchain

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: Symbol | bool
        var = Continuous()
        symb = var.create(name, min_val, max_val, precision, distribution, encoding_mechanism)
        if name not in self.variables_dict:
            self.variables_dict[name] = var
            ret = symb
        else:
            print("ERROR: the variable name is not available\n")
            ret = False
        return ret

    def add_continuous_variables_array(
        self,
        name: str,
        shape: list[int],
        min_val: float,
        max_val: float,
        precision: float = 0.0,
        distribution: str = "uniform",
        encoding_mechanism: str = "",
    ) -> np.ndarray[Any, np.dtype[Any]] | bool:
        """function for adding a continuous variable array in the variable list

        Keyword arguments:
        name -- string containing the variable name
        shape -- tuple containing the size of the array (for the current moment 3D vectors are the maximum supported)
        min_val -- float indicating the lower bound of the variable range
        max_val -- float indicating the upper bound of the variable range
        precision -- integer indicating the number of elements for sampling is a non-uniform distribution
                     is assumed and a float indicating the distance among two representable values if a
                     uniform distribution is assumed
        distribution -- string indicating the expected distribution of the value in the range
        encoding_mechanism -- string for forcing a specific encoding mechanism (prevelently for debugging purpose)
                                if empty the best encoding mechanism is decided by the toolchain

        Return values:
        symb - variable symbol or False if the variable name is not available
        """
        ret: np.ndarray[Any, np.dtype[Any]] | bool
        arr: list[Any] = []
        symb_arr: list[Any] = []
        if len(shape) == 1:
            for i in range(shape[0]):
                arr.append(Continuous())
                symb_arr.append(
                    arr[i].create(name + "_" + format(i), min_val, max_val, precision, distribution, encoding_mechanism)
                )
        elif len(shape) == 2:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append(Continuous())
                    symb_arr[i].append(
                        arr[i][j].create(
                            name + "_" + format(i) + "_" + format(j),
                            min_val,
                            max_val,
                            precision,
                            distribution,
                            encoding_mechanism,
                        )
                    )
        elif len(shape) == 3:
            for i in range(shape[0]):
                arr.append([])
                symb_arr.append([])
                for j in range(shape[1]):
                    arr[i].append([])
                    symb_arr[i].append([])
                    for k in range(shape[2]):
                        arr[i][j].append(Continuous())
                        symb_arr[i][j].append(
                            arr[i][j][k].create(
                                name + "_" + format(i) + "_" + format(j) + "_" + format(k),
                                min_val,
                                max_val,
                                precision,
                                distribution,
                                encoding_mechanism,
                            )
                        )
        else:
            print("Arrays of dimension higher than 3 are currently not supported\n")
            ret = False

        if name not in self.variables_dict:
            self.variables_dict[name] = arr
            ret = np.array(symb_arr)
        else:
            print("ERROR: the variable name is not available\n")
            ret = False

        return ret

    def move_to_binary(
        self, constraints: list[tuple[str, bool, bool, bool]], i: int = 0, letter: str = "b"
    ) -> tuple[list[tuple[str, bool, bool, bool]], int]:
        """function for writing all the declared variables
        Binary object of the pyqubo library

        Keyword arguments:
        constraints  -- constraint object for eventually adding them if necessary
        i -- number of binary variables required to increment, by default equal to 0 (useful if the function is called more than one time)
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        constraints  -- constraint object
        i -- number of binary variables required
        """

        for elem in self.variables_dict.values():
            constraints, i = self._variable_binarization(constraints, i, elem, letter)

        return constraints, i

    def _variable_binarization(
        self,
        constraints: list[tuple[str, bool, bool, bool]],
        i: int,
        elem: Discrete | Continuous | Binary | list[Any],
        letter: str = "b",
    ) -> tuple[list[tuple[str, bool, bool, bool]], int]:
        """function for writing a declared variables
        binary_var object of the qubovert library

        Keyword arguments:
        constraints  -- constraint object for eventually adding them if necessary
        i -- number of binary variables required to increment, by default equal to 0 (useful if the function is called more than one time)
        elem -- is the declared variable under observation
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        constraints  -- constraint object
        i -- number of binary variables required
        """
        if not isinstance(elem, list):
            if elem.name is not self.binary_variables_name_weight.keys():
                if isinstance(elem, Binary):
                    if elem.type == "b":
                        self.binary_variables_name_weight[elem.name] = (boolean_var(letter + format(i)),)
                        i += 1
                    elif elem.type == "s":
                        # s = 2b-1
                        self.binary_variables_name_weight[elem.name] = (
                            boolean_var(letter + format(i)),
                            2,
                            -1,
                        )
                        i += 1
                elif elem.type == "d":
                    # For the current moment, only dictionary encoding for discrete
                    (
                        self.binary_variables_name_weight,
                        constraints,
                        i,
                    ) = elem.move_to_binary(self.binary_variables_name_weight, i, constraints, letter)
                elif elem.type == "c":
                    (
                        self.binary_variables_name_weight,
                        constraints,
                        i,
                    ) = elem.move_to_binary(self.binary_variables_name_weight, i, constraints, letter)
        else:
            for el in elem:
                constraints, i = self._variable_binarization(constraints, i, el, letter)

        return constraints, i

    def conv_var(self, v1: Variable, solution: dict[str, float]) -> float | bool:
        temp = 0.0
        if isinstance(self.binary_variables_name_weight[v1.name], list):
            for el in self.binary_variables_name_weight[v1.name]:
                if not isinstance(el, str):
                    key = next(iter(el[0].variables))
                    if key not in solution:
                        print("The variable is not found\n")
                        return False
                    if solution[key] == 1:
                        temp += el[1] if len(el) > 1 else 1
                    if len(el) == 3:
                        temp += el[2]
        elif next(iter(self.binary_variables_name_weight[v1.name][0].variables)) in solution:
            key = next(iter(self.binary_variables_name_weight[v1.name][0].variables))
            if solution[key] == 1:
                temp += (
                    self.binary_variables_name_weight[v1.name][1]
                    if len(self.binary_variables_name_weight[v1.name]) > 1
                    else 1
                )
            if len(self.binary_variables_name_weight[v1.name]) == 3:
                temp += self.binary_variables_name_weight[v1.name][2]
        else:
            print("The variable is not found\n")
            return False

        return temp

    def conv_list(self, v1: list[Variable], solution: dict[str, float]) -> list[float] | bool:
        nested_temp_list = []
        for v2 in v1:
            if not isinstance(v2, list) and v2.name in self.binary_variables_name_weight:
                temp = self.conv_var(v2, solution)
                if isinstance(temp, bool):
                    return False
                nested_temp_list.append(temp)
        return nested_temp_list

    def convert_solution_var(
        self, converted_solution: dict[str, Any], var: str, solution: dict[str, float]
    ) -> dict[str, Any] | bool:
        """function for converting a solution coming from simulated annealing"""
        converted_solution[var] = []
        for j, v in enumerate(self.variables_dict[var]):
            if isinstance(v, list):
                converted_solution[var].append([])
                for v1 in v:
                    if not isinstance(v1, list) and v1.name in self.binary_variables_name_weight:
                        temp = self.conv_var(v1, solution)
                        if isinstance(temp, bool):
                            return False
                        converted_solution[var][j].append(temp)
                    elif isinstance(v1, list):
                        nested_temp_list = self.conv_list(v1, solution)
                        if isinstance(nested_temp_list, bool):
                            return False
                        converted_solution[var][j].append(nested_temp_list)
            elif v.name in self.binary_variables_name_weight:
                temp = self.conv_var(v, solution)
                if isinstance(temp, bool):
                    return False
                converted_solution[var].append(temp)
        return converted_solution

    def converted_solution_single_var(
        self, converted_solution: dict[str, Any], var: str, solution: dict[str, float]
    ) -> dict[str, Any] | bool:
        converted_solution[var] = 0.0
        if var not in self.binary_variables_name_weight:
            print("The variable is not found\n")
            return False
        if isinstance(self.binary_variables_name_weight[var], list):
            for el in self.binary_variables_name_weight[var]:
                if not isinstance(el, str):
                    key = next(iter(el[0].variables))
                    if key not in solution:
                        print("The variable is not found\n")
                        return False
                    if solution[key] == 1:
                        converted_solution[var] += el[1] if len(el) > 1 else 1
                    if len(el) == 3:
                        converted_solution[var] += el[2]
        elif next(iter(self.binary_variables_name_weight[var][0].variables)) in solution:
            key = next(iter(self.binary_variables_name_weight[var][0].variables))
            if solution[key] == 1:
                converted_solution[var] += (
                    self.binary_variables_name_weight[var][1] if len(self.binary_variables_name_weight[var]) > 1 else 1
                )
            if len(self.binary_variables_name_weight[var]) == 3:
                converted_solution[var] += self.binary_variables_name_weight[var][2]
        else:
            print("The variable is not found\n")
            return False
        return converted_solution

    def convert_simulated_annealing_solution(self, solution: dict[str, float]) -> dict[str, Any] | bool:
        """function for converting a solution coming from simulated annealing

        Keyword arguments:
        solution -- dictionary containing binary variable-value association

        Return values:
        converted_solution -- dictionary containing the original declared variables-value association

        """
        converted_solution: dict[str, Any] = {}
        for var in self.variables_dict:
            if not isinstance(self.variables_dict[var], Variable):
                temp = self.convert_solution_var(converted_solution, var, solution)
                if isinstance(temp, bool):
                    return False
                converted_solution = temp
            else:
                temp = self.converted_solution_single_var(converted_solution, var, solution)
                if isinstance(temp, bool):
                    return False
                converted_solution = temp
        return converted_solution


class Variable:
    """parent class of variables useful
    for defining common methods of the variable object"""

    def __init__(self) -> None:
        """declariation of common variable attributes
        _name is a string identifying the variable
        symbol is the symbol associated with the variable"""
        self.name: str
        self.symbol: Symbol
        self.type: str

    def _dictionary_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        constraints: list[tuple[str, bool, bool, bool]],
        values: list[float],
        letter: str = "b",
    ) -> tuple[dict[str, Any], list[tuple[str, bool, bool, bool]], int]:
        """function for encoding a dictionary into binary variables.

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        constraints -- constraints object for evantually adding constraints
        value -- a list of float containing the value to encode
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        constraints -- constraints object for eventually adding constraints
        i --  index for the binary variable
        """
        var_sum = 0
        binary_variables_name_weight[self.name] = []
        binary_variables_name_weight[self.name].append("dictionary")
        for val in values:
            binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), val))
            var_sum += symbols(letter + format(i))
            i += 1

        # Add the needed constraint
        constraints.append((format(expand(var_sum).evalf()) + " = 1", True, False, False))
        return binary_variables_name_weight, constraints, i

    def _unitary_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        max_val: float,
        min_val: float,
        unitary_weight: float = 1,
        letter: str = "b",
    ) -> tuple[dict[str, Any], int]:
        """function for implementing the unitary encoding

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        max_value -- the maximum value to represent
        min_value -- the minimum value to represent
        unitary weight -- float for eventually weighting the unitary encoding (by default set to 1)
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        i --  index for the binary variable"""
        samples = int((max_val - min_val) / unitary_weight)
        binary_variables_name_weight[self.name] = []
        binary_variables_name_weight[self.name].append("unitary")
        for w in range(1, samples + 1):
            if w == 1:
                binary_variables_name_weight[self.name].append((
                    boolean_var(letter + format(i)),
                    w * unitary_weight,
                    min_val,
                ))
            else:
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), w * unitary_weight))
            i += 1

        return binary_variables_name_weight, i

    def _domain_well_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        max_val: float,
        min_val: float,
        constraints: list[tuple[str, bool, bool, bool]],
        unitary_weight: float = 1,
        letter: str = "b",
    ) -> tuple[dict[str, Any], list[tuple[str, bool, bool, bool]], int]:
        """function for implementing the unitary encoding

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        max_value -- the maximum value to represent
        min_value -- the minimum value to represent
        constraints -- constraints object for evantually adding constraints
        unitary weight -- float for eventually weighting the unitary encoding (by default set to 1)
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        i --  index for the binary variable"""
        samples = int((max_val - min_val) / unitary_weight)
        binary_variables_name_weight[self.name] = []
        binary_variables_name_weight[self.name].append("domain well")
        for w in range(1, samples + 1):
            if w == 1:
                binary_variables_name_weight[self.name].append((
                    boolean_var(letter + format(i)),
                    w * unitary_weight,
                    min_val,
                ))
            else:
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), w * unitary_weight))
                constraints.append((letter + format(i) + ">=" + letter + format(i - 1), True, False, False))

            i += 1

        return binary_variables_name_weight, constraints, i

    def _logarithmic_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        max_val: float,
        min_val: float,
        base: int = 2,
        lower_power: int = 0,
        letter: str = "b",
    ) -> tuple[dict[str, Any], int]:
        """function for implementing the logarithmic encoding

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        max_value -- the maximum value to represent
        min_value -- the minimum value to represent
        base  -- int indicating the basis for the logarithmic encoding (set at 2 by default)
        lower_power -- int indicating the lower power of the base to consider (set at 0 by default), but useful for floating input or for regulating the precision
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        i --  index for the binary variable"""
        if lower_power == 0:
            n_power = floor(log((max_val - min_val + 1), base))
        else:
            n_power = floor(log(((max_val - min_val) / base**lower_power), base))
        binary_variables_name_weight[self.name] = []
        binary_variables_name_weight[self.name].append("logarithmic")
        binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), base**lower_power, min_val))
        i += 1
        for power in range(lower_power + 1, n_power + lower_power):
            binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), base**power))
            i += 1
        if log(((max_val - min_val + 1) / base**lower_power), base) > n_power:
            val = max_val - min_val
            for j in range(lower_power, n_power + lower_power):
                val -= base**j
            binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), val))
            i += 1

        return binary_variables_name_weight, i

    def _arithmetic_progression_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        max_val: float,
        min_val: float,
        unitary_weight: float = 1,
        letter: str = "b",
    ) -> tuple[dict[str, Any], int]:
        """function for implementing the arthmetic progression encoding

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        max_value -- the maximum value to represent
        min_value -- the minimum value to represent
        unitary weight -- float for eventually weighting the unitary encoding (by default set to 1)
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        i --  index for the binary variable"""
        n = (max_val - min_val) / unitary_weight
        samples = ceil(0.5 * sqrt(1 + 8 * n) - 0.5)
        binary_variables_name_weight[self.name] = []
        binary_variables_name_weight[self.name].append("arithmetic progression")
        for w in range(1, samples):
            if w == 1:
                binary_variables_name_weight[self.name].append((
                    boolean_var(letter + format(i)),
                    w * unitary_weight,
                    min_val,
                ))
            else:
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), w * unitary_weight))
            i += 1
        val = unitary_weight * (n - (samples * (samples - 1)) / 2)
        if samples != 1:
            binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), val))
        else:
            binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), val, min_val))
        i += 1

        return binary_variables_name_weight, i

    def _bounded_coefficient_encoding(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        ux: int,
        max_val: float,
        min_val: float,
        base: int = 2,
        lower_power: int = 0,
        letter: str = "b",
    ) -> tuple[dict[str, Any], int]:
        """function for implementing the arthmetic progression encoding

        Keyword arguments:
        binary_variables_name_weight -- dictionary in which inserting binary variables, weights and eventual offset
        i -- starting index for the binary variable
        ux -- int is the coefficients upper bound
        max_value -- the maximum value to represent
        min_value -- the minimum value to represent
        base  -- int indicating the basis for the logarithmic encoding (set at 2 by default)
        lower_power -- int indicating the lower power of the base to consider (set at 0 by default), but useful for floating input or for regulating the precision
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        i --  index for the binary variable"""
        if (max_val - min_val) < base ** (floor(log(ux, base)) + 1):
            binary_variables_name_weight, i = self._logarithmic_encoding(
                binary_variables_name_weight, i, max_val, min_val, base, lower_power
            )

            ret = binary_variables_name_weight, i
        else:  # To verify
            ro = floor(log(ux, base)) + 1
            v = max_val - min_val
            for j in range(ro):
                v -= base**j
            eta = floor(v / ux)
            binary_variables_name_weight[self.name] = []
            binary_variables_name_weight[self.name].append("bounded coefficient")
            binary_variables_name_weight[self.name].append((
                boolean_var(letter + format(i)),
                base**lower_power,
                min_val,
            ))
            i += 1

            for k in range(lower_power + 1, ro):
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), base**k))
                i += 1

            for _ in range(ro + 1, ro + eta + 1):
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), ux))
                i += 1

            if v - eta * ux != 0:
                binary_variables_name_weight[self.name].append((boolean_var(letter + format(i)), v - eta * ux))
                i += 1

            ret = binary_variables_name_weight, i
        return ret


class Binary(Variable):
    """child class of binary variables"""

    def create(self, name: str, unipolar: bool = True) -> Symbol:
        """function for creating the new binary variable.

        Keyword arguments:
        name -- string containing the variable name
        unipolar -- boolean variable equal to true if the variable is assumes 0 1 and
                    equal to false if assume -1, 1

        Return values:
        self.symbol -- variable symbol
        """
        self.name = name
        self.symbol = symbols(name)
        if unipolar:
            self.type = "b"
        else:
            self.type = "s"
        return self.symbol


class Discrete(Variable):
    """child class of discrete variables for representing variables that can assume a finite number of values"""

    def create(self, name: str, values: list[float]) -> Symbol:
        """function for creating the new discrete variable.

        Keyword arguments:
        name -- string containing the variable name
        values -- list of values that the variable can assumes

        Return values:
        self.symbol -- variable symbol
        """
        self.name = name
        self.symbol = symbols(name)
        self._values = values
        self.type = "d"
        return self.symbol

    def move_to_binary(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        constraints: list[tuple[str, bool, bool, bool]],
        letter: str = "b",
    ) -> tuple[dict[str, Any], list[tuple[str, bool, bool, bool]], int]:
        """function for creating the new discrete variable.

        Keyword arguments:
        binary_variables_name_weight -- dictionary of all the binary variables
        i -- int indicating the binary variable index
        constraints -- constraints object for eventually adding constraints
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        constraints -- constraints object for evantually adding constraints
        i --  index for the binary variable


        I start with only the dictionary mapping methods
        """
        binary_variables_name_weight, constraints, i = self._dictionary_encoding(
            binary_variables_name_weight, i, constraints, self._values, letter
        )
        return binary_variables_name_weight, constraints, i


class Continuous(Variable):
    """child class of discrete variables"""

    def create(
        self,
        name: str,
        min_val: float,
        max_val: float,
        precision: float,
        distribution: str = "uniform",
        encoding_mechanism: str = "",
    ) -> Symbol:
        """function for creating the new discrete variable.

        Keyword arguments:
        name -- string containing the variable name
        min_val -- float indicating the lower bound of the variable range
        max_val -- float indicating the upper bound of the variable range
        precision -- integer indicating the number of elements for sampling is a non-uniform distribution
                     is assumed and a float indicating the distance among two representable values if a
                     uniform distribution is assumed
        distribution -- string indicating the expected distribution of the value in the range
        encoding_mechanism -- string for forcing a specific encoding mechanism (prevelently for debugging purpose)
                                if empty the best encoding mechanism is decided by the toolchain

        Return values:
        self.symbol -- variable symbol
        """
        self.name = name
        self.symbol = symbols(name)
        self._min = min_val
        self._max = max_val
        self.precision = precision
        self._distribution = distribution
        self.type = "c"
        self._encoding_mechanism = encoding_mechanism
        return self.symbol

    def move_to_binary(
        self,
        binary_variables_name_weight: dict[str, Any],
        i: int,
        constraints: list[tuple[str, bool, bool, bool]],
        letter: str = "b",
    ) -> tuple[dict[str, Any], list[tuple[str, bool, bool, bool]], int]:
        """function for creating the new discrete variable.

        Keyword arguments:
        binary_variables_name_weight -- dictionary of all the binary variables
        i -- int indicating the binary variable index
        constraints -- constraints object for eventually adding constraints
        letter  -- letter chosen for labeling the variable b by default by could be different for auxiliary variables

        Return values:
        binary_variables_name_weight -- dictionary
        constraints -- constraints object for evantually adding constraints
        i --  index for the binary variable

        possible to expand the supported distributions in the future

        """
        if self._distribution == "logarithmic":
            values = list(np.logspace(self._min, self._max, int(self.precision), endpoint=True))
            binary_variables_name_weight, constraints, i = self._dictionary_encoding(
                binary_variables_name_weight, i, constraints, values, letter
            )
            self._encoding_mechanism = "dictionary"
        elif self._distribution == "geometric":
            values = list(np.geomspace(self._min, self._max, int(self.precision), endpoint=True))
            binary_variables_name_weight, constraints, i = self._dictionary_encoding(
                binary_variables_name_weight, i, constraints, values, letter
            )
            self._encoding_mechanism = "dictionary"
        elif self._encoding_mechanism == "dictionary":
            values = list(np.arange(self._min, self._max + self.precision, self.precision))
            binary_variables_name_weight, constraints, i = self._dictionary_encoding(
                binary_variables_name_weight, i, constraints, values, letter
            )
        elif self._encoding_mechanism == "unitary":
            binary_variables_name_weight, i = self._unitary_encoding(
                binary_variables_name_weight, i, self._max, self._min, self.precision, letter
            )
        elif self._encoding_mechanism == "domain well":
            binary_variables_name_weight, constraints, i = self._domain_well_encoding(
                binary_variables_name_weight, i, self._max, self._min, constraints, self.precision, letter
            )
        elif self._encoding_mechanism.startswith("logarithmic"):
            try:
                base = int(self._encoding_mechanism.split(" ")[1])
                binary_variables_name_weight, i = self._logarithmic_encoding(
                    binary_variables_name_weight, i, self._max, self._min, base, int(self.precision), letter
                )
                self.precision = base**self.precision
            except TypeError:
                values = list(np.arange(self._min, self._max + self.precision, self.precision))
                binary_variables_name_weight, constraints, i = self._dictionary_encoding(
                    binary_variables_name_weight, i, constraints, values, letter
                )
                self._encoding_mechanism = "dictionary"

        elif self._encoding_mechanism == "arithmetic progression":
            binary_variables_name_weight, i = self._arithmetic_progression_encoding(
                binary_variables_name_weight, i, self._max, self._min, self.precision, letter
            )
        elif self._encoding_mechanism.startswith("bounded coefficient"):
            try:
                ux = int(self._encoding_mechanism.split(" ")[2])
                binary_variables_name_weight, i = self._bounded_coefficient_encoding(
                    binary_variables_name_weight,
                    i,
                    ux,
                    self._max,
                    self._min,
                    2,
                    int(log2(self.precision)),
                    letter,
                )
            except TypeError:
                values = list(np.arange(self._min, self._max + self.precision, self.precision))
                binary_variables_name_weight, constraints, i = self._dictionary_encoding(
                    binary_variables_name_weight, i, constraints, values, letter
                )
                self._encoding_mechanism = "dictionary"
        elif (1 / self.precision) % 2 == 0:
            binary_variables_name_weight, i = self._logarithmic_encoding(
                binary_variables_name_weight, i, self._max, self._min, 2, int(log2(self.precision)), letter
            )
            self._encoding_mechanism = "logarithmic 2"
        else:
            binary_variables_name_weight, i = self._arithmetic_progression_encoding(
                binary_variables_name_weight, i, self._max, self._min, self.precision, letter
            )
            self._encoding_mechanism = "arithmetic progression"

        return binary_variables_name_weight, constraints, i
