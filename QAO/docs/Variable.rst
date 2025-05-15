Variables Class
===============

It manages the optimization problems variables.

Variables types supported
-------------------------
Three types of variables are supported:

- Binary variables:
    - unipolar, assuming 0 or 1 values
    - bipolar, assuming -1 or 1 values
- Discrete variables, assuming a user-provided list of values
- Continuous variables, assuming all values in a user-provided range with a specified step

Variables declarations
----------------------
The class provides methods to declare variables:

- *add_binary_variable(name: str)*, for adding binary variables
    - *name*: variable name
- *add_binary_variable_array(name: str, shape: list[int])*, for adding array of binary variables (supports 1, 2, and 3 dimensional arrays of variables)
    - *name*: variable name
    - *shape*: list of integers representing the shape of the array
- *add_spin_variable(name: str)*, for adding bipolar binary variables
    - *name*: variable name
- *add_spin_variable_array(name: str, shape: list[int])*, for adding array of bipolar binary variables (supports 1, 2, and 3 dimensional arrays of variables).
    - *name*: variable name
    - *shape*: list of integers representing the shape of the array
- *add_discrete_variable(name: str, values: list[float])*, for adding discrete variables. It needs the set of values.
    - *name*: variable name
    - *values*: list of values that the variable can assume
- *add_discrete_variable_array(name: str, values: list[float], shape: list[int])*, for adding array of discrete variables. Consider for all the same set of values (supports 1, 2, and 3 dimensional arrays of variables).
    - *name*: variable name
    - *values*: list of values that the variable can assume
    - *shape*: list of integers representing the shape of the array
- *add_continuous_variable(name: str, min_val: float, max_val: float, precision: float, distribution: str = "uniform", encoding_mechanism: str = "")*, for adding continuous variables. It needs the min, max values and the wanted precision.
    - *name*: variable name
    - *values*: list of values
    - *min_val*: minimum value that the variable can assume
    - *max_val*: maximum value that the variable can assume
    - *precision*: precision of the values that the variable can assume. If logarithmic encoding is considered the values must the wanted power of the base of the logarithm.
    - *distribution*: distribution of the values in the range. It can be one among:
        - *uniform*
        - *geometric*
        - *logarithmic*
    - *encoding_mechanism*: encoding mechanism for the variable. By default logarithmic encoding is chosen if the precision is a power of two, arithmetic progression otherwise. It can be one among:
        - *dictionary*
        - *unitary*
        - *logarithmic _base_*, with the possibility of specifying the base of the logarithm. Two is considered as default.
        - *arithmetic progression*
        - *bounded coefficient _bound_*, where _bound_ is the maximum coefficient that can be used in the encoding.
- *add_continuous_variable_array(name: str, min_val: float, max_val: float, precision: float, distribution: str = "uniform", encoding_mechanism: str = "", shape: list[int])*, for adding array of continuous variables. Consider for all the same min, max values and the wanted precision (supports 1, 2, and 3 dimensional arrays of variables).
    - *name*: variable name
    - *values*: list of values
    - *min_val*: minimum value that the variable can assume
    - *max_val*: maximum value that the variable can assume
    - *precision*: precision of the values that the variable can assume. If logarithmic encoding is considered the values must the wanted power of the base of the logarithm.
    - *distribution*: distribution of the values in the range. It can be one among:
        - *uniform*
        - *geometric*
        - *logarithmic*
    - *encoding_mechanism*: encoding mechanism for the variable.  By default logarithmic encoding is chosen if the precision is a power of two, arithmetic progression otherwise. It can be one among:
        - *dictionary*
        - *unitary*
        - *logarithmic _base_*, with the possibility of specifying the base of the logarithm. Two is considered as default.
        - *arithmetic progression*
        - *bounded coefficient _bound_*, where _bound_ is the maximum coefficient that can be used in the encoding.
    - *shape*: list of integers representing the shape of the array

Example:
--------
.. code-block:: python

    from mqt.qao.variables import Variables

    # Variables object declaration
    variables = Variables()

    # declaration of a unipolar binary variable
    a0 = variables.add_binary_variable("a")
    # declaration of a discrete variable, which can assume values -1, 1, 3
    b0 = variables.add_discrete_variable("b", [-1, 1, 3])
    # declaration of a continuous variable, which can assume values in the range [-2, 2] with a precision of 0.25
    c0 = variables.add_continuous_variable("c", -2, 2, 0.25)

    # declaration of a 2D array of continuous variables in the range [-1, 1] with a precision of 0.5
    m1 = variables.add_continuous_variables_array(
        "M1", [1, 2], -1, 2, -1, "uniform", "logarithmic 2"
    )
    m2 = variables.add_continuous_variables_array(
        "M2", [2, 1], -1, 2, -1, "uniform", "logarithmic 2"
    )
