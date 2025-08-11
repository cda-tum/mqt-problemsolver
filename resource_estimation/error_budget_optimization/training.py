from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from collections import OrderedDict

    from numpy.typing import NDArray


def process_data(data: list[OrderedDict[str, float | int]]) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Splits the input data into training and testing sets.

    The function separates the features (X) and targets (Y) from the input data,
    then randomly shuffles the data and splits it into training and testing sets
    using a 75/25 ratio.

    Args:
        data: A list where the last three columns are considered targets (Y)
            and the remaining columns are features (X).

    Returns:
        X_train: Training set features.
        X_test: Testing set features.
        Y_train: Training set targets.
        Y_test: Testing set targets.
    """
    # Transform list of OrderedDicts to a NumPy array of values
    data_array = np.array([list(d.values()) for d in data])
    X = data_array[:, :-3]
    Y = data_array[:, -3:]

    np.random.seed(142)
    indices = np.random.permutation(len(X))

    train_size = int(0.75 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test


def train(data: NDArray) -> tuple[RandomForestRegressor, NDArray, NDArray]:
    """
    Trains a Random Forest Regressor on the provided data.

    The function processes the input data to separate features and targets,
    splits the data into training and testing sets, and then trains a
    Random Forest Regressor on the training set.

    Args:
        data: A NumPy array where the last three columns are considered targets (Y)
            and the remaining columns are features (X).

    Returns:
        model: The trained Random Forest Regressor.
        X_test: Testing set features.
        Y_test: Testing set targets.
    """
    X_train, X_test, Y_train, Y_test = process_data(data)

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    return model, X_test, Y_test
