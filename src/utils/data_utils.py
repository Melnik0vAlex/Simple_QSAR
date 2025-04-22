"""
data_utils.py

Вспомогательные функции для загрузки, фильтрации и сохранения данных в QSAR-проекте.
"""

import os
import logging
from typing import Dict, Union
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def load_ini_data_from_csv(
    file_name: str,
    column_name: str,
    separator: str = ",",
    delete_duplicates: bool = True,
    smiles_column: str = "SMILES",
) -> pd.DataFrame:
    """
    Загружает исходные данные из CSV-файла и выполняет валидацию.

    Args:
        file_name: Путь к CSV-файлу.
        column_name: Название целевой переменной.
        separator: Разделитель (по умолчанию ',').
        delete_duplicates: Удалять дубликаты по SMILES.
        smiles_column: Название колонки с SMILES.

    Returns:
        pd.DataFrame: Очищенный датафрейм.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Файл '{file_name}' не найден.")

    try:
        data = pd.read_csv(file_name, encoding="utf-8", sep=separator)
    except Exception as e:
        logger.error("Ошибка загрузки CSV: %s", e)
        raise ValueError(f"Не удалось загрузить CSV-файл: {e}") from e

    required = [smiles_column, column_name]
    missing = [col for col in required if col not in data.columns]
    if missing:
        logger.error("Отсутствуют обязательные колонки: %s", missing)
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    logger.info("Загружено записей: %d", len(data))

    data.dropna(subset=required, inplace=True)
    logger.info("После удаления NaN: %d", len(data))

    if delete_duplicates:
        before = len(data)
        data.drop_duplicates(subset=[smiles_column], inplace=True)
        logger.info("Удалено дубликатов: %d", before - len(data))

    return data.reset_index(drop=True)


def save_descriptors_to_csv(descriptors: pd.DataFrame, file_path: str) -> None:
    """
    Сохраняет дескрипторы в CSV.

    Args:
        descriptors: Таблица дескрипторов.
        file_path: Путь к целевому файлу.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        descriptors.to_csv(file_path, sep=",", encoding="utf-8", index=False)
        logger.info("Дескрипторы сохранены в %s", file_path)
    except Exception as e:
        logger.error("Ошибка при сохранении файла: %s", e)
        raise


def check_existing_smiles_to_dataframe(
    check_smiles_file: str,
    training_data_path: str,
    smiles_column: str = "SMILES",
    target_column: str = "target",
) -> pd.DataFrame:
    """
    Возвращает только новые SMILES (не встречавшиеся в обучении).

    Args:
        check_smiles_file: Файл с SMILES и значениями для проверки.
        training_data_path: Обучающий CSV-файл.
        smiles_column: Название колонки с SMILES.
        target_column: Название целевой переменной.

    Returns:
        pd.DataFrame: Фильтрованный датафрейм с новыми молекулами.
    """
    training_data = pd.read_csv(training_data_path)

    if smiles_column not in training_data.columns:
        raise ValueError(f"Колонка '{smiles_column}' отсутствует в обучающих данных.")

    # Загрузка файла для проверки
    if check_smiles_file.endswith(".csv"):
        check_data = pd.read_csv(check_smiles_file)
        if (
            smiles_column not in check_data.columns
            or target_column not in check_data.columns
        ):
            raise ValueError(
                f"Файл должен содержать '{smiles_column}' и '{target_column}'."
            )
    else:
        raise ValueError("Формат файла не поддерживается. Используйте CSV.")

    def canonicalize(smiles: str) -> Union[str, None]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except (ValueError, TypeError, AttributeError) as e:
            logging.debug("Ошибка канонизации SMILES '%s': %s", smiles, e)
            return None

    training_smiles = {
        canonicalize(s) for s in training_data[smiles_column].dropna().unique()
    }

    check_data["Canonical_SMILES"] = check_data[smiles_column].apply(canonicalize)
    filtered = check_data[check_data["Canonical_SMILES"].notna()].copy()

    filtered.loc[:, "Status"] = filtered["Canonical_SMILES"].apply(
        lambda x: "existing" if x in training_smiles else "not_existing"
    )

    result = filtered[filtered["Status"] == "not_existing"][
        [smiles_column, target_column]
    ]

    logger.info("Отобрано новых молекул: %d", len(result))
    return result.reset_index(drop=True)


def smiles_dict_to_dataframe(
    smiles_for_prediction: Dict[str, list], target_column: str = "target"
) -> pd.DataFrame:
    """
    Преобразует словарь со списками SMILES в DataFrame со статусом.

    Args:
        smiles_for_prediction: {'existing': [...], 'not_existing': [...]}
        target_column: Название колонки под будущие значения (пока None).

    Returns:
        pd.DataFrame: Таблица с колонками 'SMILES', target_column и 'Status'.
    """
    smiles_list = []

    for smi in smiles_for_prediction.get("existing", []):
        smiles_list.append({"SMILES": smi, target_column: None, "Status": "existing"})

    for smi in smiles_for_prediction.get("not_existing", []):
        smiles_list.append(
            {"SMILES": smi, target_column: None, "Status": "not_existing"}
        )

    return pd.DataFrame(smiles_list)
