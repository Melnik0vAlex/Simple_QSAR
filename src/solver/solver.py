"""
solver.py

Модуль для построения и оценки QSAR-моделей на основе различных стратегий подготовки данных,
включая подбор гиперпараметров и сохранение модели.
"""

import logging
from typing import Optional, Tuple, Dict
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from descriptors.descriptors_api import compute_all_descriptors
from models.regressionmodel import RegressionModel
from utils.statfunctions import (
    remove_highly_correlated_descriptors,
    select_significant_descriptors,
)


def run_qsar_pipeline_from_smiles(
    data: pd.DataFrame,
    descriptors_file_path: str,
    target_column: str,
    model_name: str = "LinearRegression",
    correlation_threshold: float = 0.9,
    scale_data: bool = True,
    plotting_graph: bool = True,
    save_model: Optional[str] = None,
    grid_search_params: Optional[Dict] = None,
) -> RegressionModel:
    """
    Построение QSAR-модели с вычислением дескрипторов по SMILES.

    Args:
        data (pd.DataFrame): Данные с колонкой 'SMILES' и целевой переменной.
        descriptors_file_path (str): Путь к файлу для сохранения/загрузки дескрипторов.
        target_column (str): Название целевого столбца (например, 'pIC50').
        model_name (str): Название модели.
        correlation_threshold (float): Порог корреляции.
        scale_data (bool): Масштабировать ли данные.
        plotting_graph (bool): Строить ли график.
        save_model (str, optional): Путь к файлу для сохранения модели.
        grid_search_params (dict, optional): Параметры для подбора гиперпараметров.

    Returns:
        RegressionModel: Обученная модель.
    """
    logging.info("Запуск QSAR-пайплайна с дескрипторами.")
    data_with_descriptors = compute_all_descriptors(data, descriptors_file_path)

    if target_column not in data_with_descriptors.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных.")

    feature_columns = data_with_descriptors.columns.difference(
        ["SMILES", target_column]
    )
    x = data_with_descriptors[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = data_with_descriptors[target_column].dropna()

    x.fillna(x.mean(), inplace=True)
    x, y = x.align(y, join="inner", axis=0)

    # Удаление коррелированных дескрипторов
    corr_matrix = x.corr().abs()
    selected = remove_highly_correlated_descriptors(
        corr_matrix, threshold=correlation_threshold
    )
    x = x[selected]

    # Отбор значимых дескрипторов
    x, _ = select_significant_descriptors(x, y, alpha=0.05)
    if x.empty:
        raise ValueError("После отбора не осталось значимых дескрипторов.")

    # Масштабирование
    if scale_data:
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)

    # Обучение модели
    model = RegressionModel()
    model.select_model(model_name)
    model.prepare_data(x, y)

    if grid_search_params:
        best_params = perform_grid_search(model, x, y, grid_search_params)
        print(f"🏁 Лучшие параметры для модели {model_name}: {best_params}")

    model.train()
    metrics = model.evaluate_descriptive_metrics()
    cv = model.cross_validate(x, y)

    # Результаты
    print(f"\nМодель: {model_name}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Кросс-валидация (R²): {cv['Mean']:.4f} ± {cv['Std']:.4f}")

    if plotting_graph:
        model.plot_model_performance(column_name=target_column)

    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"💾 Модель сохранена в файл: {save_model}")

    return model


def run_qsar_pipeline_from_descriptors(
    file_path: str,
    target_column: str,
    model_name: str = "LinearRegression",
    correlation_threshold: float = 0.9,
    scale_data: bool = True,
    plotting_graph: bool = True,
    save_model: Optional[str] = None,
    grid_search_params: Optional[Dict] = None,
) -> RegressionModel:
    """
    QSAR-пайплайн с заранее рассчитанными дескрипторами.

    Args:
        file_path (str): Путь к CSV-файлу.
        target_column (str): Целевая переменная.
        model_name (str): Название модели.
        correlation_threshold (float): Порог корреляции.
        scale_data (bool): Масштабировать ли данные.
        plotting_graph (bool): Строить ли график.
        save_model (str, optional): Сохранить модель.
        grid_search_params (dict, optional): Гиперпараметры.

    Returns:
        RegressionModel: Обученная модель.
    """
    data = pd.read_csv(file_path)
    if target_column not in data.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена.")

    x = data.drop(columns=["SMILES", target_column], errors="ignore")
    y = data[target_column].dropna()
    x = x.apply(pd.to_numeric, errors="coerce")
    x.fillna(x.mean(), inplace=True)
    x, y = x.align(y, axis=0)

    corr_matrix = x.corr().abs()
    selected = remove_highly_correlated_descriptors(
        corr_matrix, threshold=correlation_threshold
    )
    x = x[selected]

    x, _ = select_significant_descriptors(x, y, alpha=0.05)

    if scale_data:
        x = pd.DataFrame(
            StandardScaler().fit_transform(x), columns=x.columns, index=x.index
        )

    model = RegressionModel()
    model.select_model(model_name)
    model.prepare_data(x, y)

    if grid_search_params:
        best_params = perform_grid_search(model, x, y, grid_search_params)
        print(f"🏁 Лучшие параметры для модели {model_name}: {best_params}")

    model.train()
    metrics = model.evaluate_descriptive_metrics()
    cv = model.cross_validate(x, y)

    print(f"\nМодель: {model_name}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Кросс-валидация (R²): {cv['Mean']:.4f} ± {cv['Std']:.4f}")

    if plotting_graph:
        model.plot_model_performance(column_name=target_column)

    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"💾 Модель сохранена в файл: {save_model}")

    return model


def perform_grid_search(
    model: RegressionModel,
    x: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict,
    cv: int = 5,
    scoring: str = "r2",
    n_jobs: int = -1,
) -> Dict:
    """
    Подбор гиперпараметров с помощью GridSearchCV.

    Args:
        model (RegressionModel): Обёртка модели.
        x (pd.DataFrame): Данные.
        y (pd.Series): Целевые значения.
        param_grid (dict): Сетка параметров.
        cv (int): Кол-во фолдов.
        scoring (str): Метрика.
        n_jobs (int): Параллельные процессы.

    Returns:
        dict: Лучшие параметры.
    """
    if model.current_model is None:
        raise ValueError("Модель не выбрана. Используйте select_model().")

    logging.info("Запуск GridSearchCV для модели %s", model.current_model_name)
    grid = GridSearchCV(
        estimator=model.current_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=2,
    )
    grid.fit(x, y)

    model.models[model.current_model_name] = grid.best_estimator_
    model.select_model(model.current_model_name)
    print(f"🏁 Лучшие параметры: {grid.best_params_}")
    return grid.best_params_
