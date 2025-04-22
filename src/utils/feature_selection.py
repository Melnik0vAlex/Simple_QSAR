"""
feature_selection.py

Функции для отбора и оценки признаков в QSAR-моделировании.
"""

import logging
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def remove_highly_correlated_descriptors(
    corr_matrix: pd.DataFrame, threshold: float = 0.9
) -> List[str]:
    """
    Удаляет сильно коррелированные дескрипторы, оставляя по одному из каждой группы.

    Args:
        corr_matrix (pd.DataFrame): Матрица корреляций дескрипторов (обычно abs(df.corr())).
        threshold (float): Порог корреляции, выше которого дескрипторы считаются избыточными.

    Returns:
        List[str]: Список оставшихся дескрипторов.
    """
    columns = np.array(corr_matrix.columns)
    corr_array = np.abs(corr_matrix.values)
    to_drop = np.zeros(len(columns), dtype=bool)

    for i in range(len(columns)):
        if to_drop[i]:
            continue
        for j in range(i + 1, len(columns)):
            if corr_array[i, j] > threshold:
                to_drop[j] = True

    return columns[~to_drop].tolist()


def select_significant_descriptors(
    X: pd.DataFrame,  # pylint: disable=invalid-name
    y: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
    handle_nan: str = "drop",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Отбирает значимые дескрипторы на основе p-value (F-регрессия).

    Args:
        X (pd.DataFrame): Матрица дескрипторов.
        y (pd.Series or np.ndarray): Целевая переменная.
        alpha (float): Уровень значимости для отбора признаков.
        handle_nan (str): Стратегия обработки NaN: 'drop' или 'fill'.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - DataFrame с выбранными дескрипторами.
            - Список их имён.
    """
    if X.shape[0] != len(y):
        raise ValueError("Размерности X и y не совпадают.")

    if handle_nan == "drop":
        X_cleaned = X.dropna(axis=0)  # pylint: disable=invalid-name
        y_cleaned = pd.Series(y).loc[X_cleaned.index]
    elif handle_nan == "fill":
        X_cleaned = X.fillna(X.mean())  # pylint: disable=invalid-name
        y_cleaned = y
    else:
        raise ValueError("handle_nan должен быть 'drop' или 'fill'.")

    f_stats, p_values = f_regression(X_cleaned, y_cleaned)

    logging.info("Результаты F-регрессии (p-value):")
    for col, f_stat, p_val in zip(X_cleaned.columns, f_stats, p_values):
        logging.info("Дескриптор: %s | F: %.4f | p: %.4e", col, f_stat, p_val)

    mask = p_values < alpha
    selected_df = X_cleaned.loc[:, mask]
    selected_cols = X_cleaned.columns[mask].tolist()

    logging.info("Отобрано значимых дескрипторов: %d", len(selected_cols))

    return selected_df, selected_cols


def evaluate_prediction(true_value: float, predicted_value: float) -> dict:
    """
    Оценивает точность предсказания по ряду метрик.

    Args:
        true_value (float): Истинное значение.
        predicted_value (float): Предсказанное значение.

    Returns:
        dict: Метрики: MAPE, MAE, MSE, RMSE.
    """
    if true_value == 0:
        raise ValueError("Истинное значение не должно быть равно нулю.")

    y_true = np.array([true_value])
    y_pred = np.array([predicted_value])

    mape = abs((true_value - predicted_value) / true_value) * 100
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAPE": mape,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
