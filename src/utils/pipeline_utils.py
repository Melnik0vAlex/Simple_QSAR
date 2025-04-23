"""
pipeline_utils.py

Модуль для предобработки данных, масштабирования, а также
сохранения и загрузки моделей и масштабаторов в рамках QSAR-пайплайна.
"""

import logging
from pathlib import Path
from typing import Tuple, Union, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def clean_and_align_data(
    x: pd.DataFrame, y: Union[pd.Series, np.ndarray]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Очистка и синхронизация признаков и целевой переменной.

    - Преобразует X в числовой формат.
    - Заполняет пропуски средними значениями.
    - Удаляет строки с NaN в y и синхронизирует X и y по индексам.

    Args:
        x (pd.DataFrame): Матрица признаков.
        y (pd.Series or np.ndarray): Целевая переменная.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Обновлённые X и y.
    """
    x = x.apply(pd.to_numeric, errors="coerce")
    x.fillna(x.mean(), inplace=True)

    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=x.index)

    y = y.dropna()
    x, y = x.align(y, join="inner", axis=0)

    logging.info(
        "Данные очищены и синхронизированы. Размеры: X=%s, y=%s", x.shape, y.shape
    )
    return x, y


def scale_features(x: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Масштабирует данные с использованием StandardScaler.

    Args:
        x (pd.DataFrame): Матрица признаков.
        save_path (str, optional): Путь для сохранения обученного масштабатора.

    Returns:
        pd.DataFrame: Масштабированная матрица признаков.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        logging.info("Масштабатор сохранён в файл: %s", save_path)

    return pd.DataFrame(x_scaled, columns=x.columns, index=x.index)


def save_model(model, file_path: str):
    """
    Сохраняет модель на диск с использованием joblib.

    Args:
        model: Объект модели (например, RegressionModel или sklearn-модель).
        file_path (str): Путь к файлу.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logging.info("Модель сохранена в файл: %s", file_path)


def load_model(file_path: str):
    """
    Загружает ранее сохранённую модель.

    Args:
        file_path (str): Путь к файлу с моделью.

    Returns:
        object: Загруженная модель.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл модели не найден: {file_path}")
    model = joblib.load(file_path)
    logging.info("Модель загружена из файла: %s", file_path)
    return model
