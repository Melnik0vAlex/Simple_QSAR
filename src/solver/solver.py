"""
solver.py

Модуль для построения и оценки QSAR-моделей на основе различных стратегий:
- построение модели с нуля по SMILES;
- использование уже рассчитанных дескрипторов;
- подбор гиперпараметров;
- сохранение обученных моделей.
"""

import logging
from typing import Optional, Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV

from descriptors.descriptors_api import compute_all_descriptors
from models.regression_model import RegressionModel
from utils.feature_selection import (
    remove_highly_correlated_descriptors,
    select_significant_descriptors,
)
from utils.pipeline_utils import (
    clean_and_align_data,
    scale_features,
    save_model,
)
from visualization.plot_utils import plot_model_performance


def run_qsar_pipeline_from_smiles(
    data: pd.DataFrame,
    descriptors_file_path: str,
    target_column: str,
    model_name: str = "LinearRegression",
    correlation_threshold: float = 0.9,
    scale_data: bool = True,
    plotting_graph: bool = True,
    save_model_path: Optional[str] = None,
    grid_search_params: Optional[Dict] = None,
) -> RegressionModel:
    """
    QSAR-пайплайн с автоматическим расчётом дескрипторов по SMILES.

    Args:
        data (pd.DataFrame): Данные с колонкой SMILES и целевым столбцом.
        descriptors_file_path (str): Путь к файлу с результатами дескрипторов.
        target_column (str): Название целевого столбца.
        model_name (str): Название модели.
        correlation_threshold (float): Порог для удаления коррелированных признаков.
        scale_data (bool): Масштабировать ли признаки.
        plotting_graph (bool): Построить график.
        save_model_path (str, optional): Путь для сохранения модели.
        grid_search_params (dict, optional): Сетка параметров для подбора.

    Returns:
        RegressionModel: Обученная модель.
    """
    logging.info("🚀 Старт пайплайна QSAR по SMILES.")
    data_with_descriptors = compute_all_descriptors(data, descriptors_file_path)

    if target_column not in data_with_descriptors.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена в данных.")

    x = data_with_descriptors.drop(columns=["SMILES", target_column], errors="ignore")
    y = data_with_descriptors[target_column]

    x, y = clean_and_align_data(x, y)

    # Удаление сильно коррелированных дескрипторов
    corr_matrix = x.corr().abs()
    x = x[
        remove_highly_correlated_descriptors(
            corr_matrix, threshold=correlation_threshold
        )
    ]

    # Отбор значимых дескрипторов
    x, _ = select_significant_descriptors(x, y)

    # Масштабирование
    if scale_data:
        x = scale_features(x)

    # Инициализация и обучение модели
    model = RegressionModel()
    model.select_model(model_name)
    model.prepare_data(x, y)

    if grid_search_params:
        best_params = perform_grid_search(model, x, y, grid_search_params)
        print(f"🔧 Лучшие параметры: {best_params}")

    model.train()

    metrics = model.evaluate_descriptive_metrics()
    cv = model.cross_validate(x, y)

    print_model_metrics(model_name, metrics, cv)

    if plotting_graph:
        plot_model_performance(
            y_train=model.y_train,
            y_pred_train=model.y_pred_train,
            y_test=model.y_test,
            y_pred_test=model.y_pred_test,
            model_name=model.current_model_name,
            column_name=target_column,
            metrics=model.evaluate_descriptive_metrics(),
        )

    if save_model_path:
        save_model(model, save_model_path)

    return model


def run_qsar_pipeline_from_descriptors(
    file_path: str,
    target_column: str,
    model_name: str = "LinearRegression",
    correlation_threshold: float = 0.9,
    scale_data: bool = True,
    plotting_graph: bool = True,
    save_model_path: Optional[str] = None,
    grid_search_params: Optional[Dict] = None,
) -> RegressionModel:
    """
    QSAR-пайплайн с использованием заранее вычисленных дескрипторов.

    Args:
        file_path (str): CSV-файл с дескрипторами.
        target_column (str): Название целевого столбца.
        model_name (str): Название модели.
        correlation_threshold (float): Порог корреляции.
        scale_data (bool): Масштабировать признаки.
        plotting_graph (bool): Построить график.
        save_model_path (str, optional): Путь для сохранения модели.
        grid_search_params (dict, optional): Гиперпараметры для подбора.

    Returns:
        RegressionModel: Обученная модель.
    """
    data = pd.read_csv(file_path)

    if target_column not in data.columns:
        raise ValueError(f"Целевая переменная '{target_column}' не найдена.")

    x = data.drop(columns=["SMILES", target_column], errors="ignore")
    y = data[target_column]

    x, y = clean_and_align_data(x, y)

    x = x[
        remove_highly_correlated_descriptors(
            x.corr().abs(), threshold=correlation_threshold
        )
    ]
    x, _ = select_significant_descriptors(x, y)

    if scale_data:
        x = scale_features(x)

    model = RegressionModel()
    model.select_model(model_name)
    model.prepare_data(x, y)

    if grid_search_params:
        best_params = perform_grid_search(model, x, y, grid_search_params)
        print(f"🔧 Лучшие параметры: {best_params}")

    model.train()
    metrics = model.evaluate_descriptive_metrics()
    cv = model.cross_validate(x, y)

    print_model_metrics(model_name, metrics, cv)

    if plotting_graph:
        model.plot_model_performance(column_name=target_column)

    if save_model_path:
        save_model(model, save_model_path)

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
    Подбор гиперпараметров с использованием GridSearchCV.

    Args:
        model (RegressionModel): Обёртка модели.
        x (pd.DataFrame): Признаки.
        y (pd.Series): Целевая переменная.
        param_grid (dict): Сетка параметров.
        cv (int): Кол-во фолдов.
        scoring (str): Метрика.
        n_jobs (int): Кол-во потоков.

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

    return grid.best_params_


def print_model_metrics(model_name: str, metrics: Dict, cv_results: Dict):
    """
    Печатает метрики модели в консоль.

    Args:
        model_name (str): Название модели.
        metrics (dict): Метрики описательной и предсказательной способности.
        cv_results (dict): Результаты кросс-валидации.
    """
    print(f"\n🧠 Результаты модели: {model_name}")
    print("📊 Описательная способность:")
    print(f"R² = {metrics['R^2']:.4f}")
    print(f"RMSE = {metrics['RMSE']:.4f}")
    print(f"Pearson r = {metrics['Pearson r']:.4f}")
    print(f"Spearman rho = {metrics['Spearman rho']:.4f}")
    print("\n📈 Предсказательная способность:")
    print(f"PRESS = {metrics['PRESS']:.4f}")
    print(f"PRMSE = {metrics['PRMSE']:.4f}")
    print(f"Q² = {metrics['Q^2']:.4f}")
    print(f"Кросс-валидация (R²): {cv_results['Mean']:.4f} ± {cv_results['Std']:.4f}")
