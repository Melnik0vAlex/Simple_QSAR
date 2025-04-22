"""
descriptors_api.py

API-функции для вычисления 1D/2D и 3D дескрипторов (batch и одиночный режим).
"""

import logging
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from descriptors.descriptors_1d2d import (
    compute_1d_2d_descriptors,
    descriptor_1d_2d_names,
)
from descriptors.descriptors_3d import (
    compute_3d_descriptors,
    descriptor_3d_names,
)


def compute_all_descriptors(
    data: pd.DataFrame,
    file_name: Optional[str] = None,
    smiles_column: str = "SMILES",
    save_to_file: bool = True,
    mode: str = "all",  # 'all', '1d2d', '3d'
) -> pd.DataFrame:
    """
    Вычисляет молекулярные дескрипторы (1D/2D и/или 3D) для каждой молекулы в таблице.

    Args:
        data (pd.DataFrame): Таблица с колонкой SMILES.
        file_name (str, optional): Имя CSV-файла для сохранения дескрипторов.
        smiles_column (str): Название колонки с SMILES.
        save_to_file (bool): Сохранять ли дескрипторы в файл.
        mode (str): 'all', '1d2d', или '3d'.

    Returns:
        pd.DataFrame: Исходные данные + рассчитанные дескрипторы.
    """
    if smiles_column not in data.columns:
        raise ValueError(f"Колонка '{smiles_column}' отсутствует в данных.")

    logging.info("🚀 Запуск расчёта дескрипторов.")
    start_time = time.time()

    # === Вычисление 1D/2D дескрипторов ===
    if mode in ("all", "1d2d"):

        def compute_1d_2d(smiles):
            try:
                return compute_1d_2d_descriptors(smiles)
            except Exception as e:
                logging.error("Ошибка 1D/2D-дескрипторов для %s: %s", smiles, str(e))
                return [None] * len(descriptor_1d_2d_names)

        logging.info("Вычисление 1D/2D дескрипторов...")
        with ThreadPoolExecutor() as executor:
            descriptor_1d_2d = list(
                tqdm(
                    executor.map(compute_1d_2d, data[smiles_column]),
                    desc="1D/2D дескрипторы",
                    total=len(data),
                    unit=" molecule",
                    bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}",
                )
            )
        descriptors_1d_2d_df = pd.DataFrame(
            descriptor_1d_2d, columns=descriptor_1d_2d_names
        )
    else:
        descriptors_1d_2d_df = pd.DataFrame()

    # === Вычисление 3D дескрипторов ===
    if mode in ("all", "3d"):

        def compute_3d(smiles):
            try:
                return compute_3d_descriptors(smiles)
            except Exception as e:
                logging.error("Ошибка 3D-дескрипторов для %s: %s", smiles, str(e))
                failed_smiles.append(smiles)
                return [None] * len(descriptor_3d_names)

        failed_smiles = []
        logging.info("🧱 Вычисление 3D дескрипторов...")
        with ThreadPoolExecutor() as executor:
            descriptor_3d = list(
                tqdm(
                    executor.map(compute_3d, data[smiles_column]),
                    desc="3D дескрипторы",
                    total=len(data),
                    unit=" molecule",
                    bar_format="{l_bar} {bar} {n_fmt}/{total_fmt}",
                )
            )
        descriptors_3d_df = pd.DataFrame(descriptor_3d, columns=descriptor_3d_names)
    else:
        descriptors_3d_df = pd.DataFrame()

    # === Объединение всех дескрипторов с исходными данными ===
    combined = pd.concat(
        [data.reset_index(drop=True), descriptors_1d_2d_df, descriptors_3d_df], axis=1
    )

    # === Сохранение результата ===
    if save_to_file and file_name:
        combined.to_csv(file_name, index=False)
        logging.info("📁 Дескрипторы сохранены в файл: %s", file_name)

    # === Сохранение ошибок ===
    if mode in ("all", "3d") and "failed_smiles" in locals() and failed_smiles:
        with open("logs/failed_smiles.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed_smiles))
        logging.warning(
            "❗ Не удалось рассчитать 3D-дескрипторы для %d молекул.",
            len(failed_smiles),
        )

    # === Завершение ===
    elapsed = time.time() - start_time
    logging.info("✅ Расчёт дескрипторов завершён за %.2f секунд.", elapsed)
    return combined


def compute_descriptors_for_smiles(smiles: str, mode: str = "all") -> pd.DataFrame:
    """
    Вычисляет дескрипторы для одного SMILES (1D/2D и/или 3D).

    Args:
        smiles (str): Строка SMILES.
        mode (str): 'all', '1d2d', или '3d'.

    Returns:
        pd.DataFrame: Строка с рассчитанными дескрипторами.
    """
    data = pd.DataFrame({"SMILES": [smiles]})
    result = compute_all_descriptors(
        data, file_name=None, save_to_file=False, mode=mode
    )
    return result.iloc[[0]]
