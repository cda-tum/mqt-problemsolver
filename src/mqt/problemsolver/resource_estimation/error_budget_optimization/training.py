from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from collections import OrderedDict

    from numpy.typing import NDArray


def _process_data(
    data: list[OrderedDict[str, float | int]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Splits the input data into training and testing sets.

    The function separates the features (x) and targets (y) from the input data,
    then randomly shuffles the data and splits it into training and testing sets
    using a 75/25 ratio.

    Args:
        data: A list where the last three columns are considered targets (y)
            and the remaining columns are features (x).

    Returns:
        x_train: Training set features.
        x_test: Testing set features.
        y_train: Training set targets.
        y_test: Testing set targets.
    """
    # Transform list of OrderedDicts to a NumPy array of values
    data_array = np.array([list(d.values()) for d in data])
    x = data_array[:, :-3]
    y = data_array[:, -3:]

    rng = np.random.default_rng(142)
    indices = rng.permutation(len(x))

    train_size = int(0.75 * len(x))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return x_train, x_test, y_train, y_test


def train(
    data: list[OrderedDict[str, float | int]],
) -> tuple[RandomForestRegressor, NDArray[np.float64], NDArray[np.float64]]:
    """Trains a Random Forest Regressor on the provided data.

    The function processes the input data to separate features and targets,
    splits the data into training and testing sets, and then trains a
    Random Forest Regressor on the training set.

    Args:
        data: A NumPy array where the last three columns are considered targets (Y)
            and the remaining columns are features (X).

    Returns:
        model: The trained Random Forest Regressor.
        x_test: Testing set features.
        y_test: Testing set targets.
    """
    x_train, x_test, y_train, y_test = _process_data(data)

    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    return model, x_test, y_test
