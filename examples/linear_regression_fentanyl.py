# -*- coding: utf-8 -*-
"""
Пример QSAR-моделирования на данных фентанила с LinearRegression.
"""
# pylint: disable=wrong-import-position
import sys
from pathlib import Path

# Добавим корень проекта для импорта модулей
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from solver.solver import (
    run_qsar_pipeline_from_smiles,
)
from utils.data_utils import (
    load_ini_data_from_csv,
)


def run_example():
    print("🚀 Запуск QSAR-моделирования (LinearRegression)...")

    data_path = "data/raw/fentanyl_smiles_target.csv"
    descriptors_path = "data/interim/fentanyl_descriptors.csv"
    model_output = "models/regressors/linear_fentanyl.pkl"

    # Загрузка исходных данных
    data = load_ini_data_from_csv(data_path, column_name="pKi", separator=",")

    # Запуск пайплайна
    run_qsar_pipeline_from_smiles(
        data=data,
        descriptors_file_path=descriptors_path,
        target_column="pKi",
        model_name="LinearRegression",
        correlation_threshold=0.9,
        scale_data=True,
        plotting_graph=True,
        save_model_path=model_output,
    )

    print("✅ QSAR-модель успешно обучена и сохранена.")


if __name__ == "__main__":
    run_example()
