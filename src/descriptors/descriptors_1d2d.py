import logging
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

# Отключение RDKit-логов
RDLogger.DisableLog("rdApp.*")

# Лог-файл для ошибок дескрипторов
logging.basicConfig(
    filename="logs/descriptors1d2d_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Список доступных дескрипторов RDKit (имена и функции)
descriptor_1d_2d_names = [desc[0] for desc in Descriptors.descList]
descriptor_1d_2d_functions = [desc[1] for desc in Descriptors.descList]


def compute_1d_2d_descriptors(smiles):
    """
    Вычисляет RDKit 1D и 2D дескрипторы для заданной молекулы по SMILES.

    Args:
        smiles (str): SMILES-строка молекулы.

    Returns:
        dict: Словарь {имя_дескриптора: значение}, или None при ошибке.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        logging.warning(
            "[SMILES ERROR] Не удалось создать молекулу из SMILES: %s", smiles
        )
        return {name: None for name in descriptor_1d_2d_names}

    descriptors = {}
    for name, func in zip(descriptor_1d_2d_names, descriptor_1d_2d_functions):
        try:
            value = func(mol)
            if isinstance(value, float) and np.isnan(value):
                value = None
            descriptors[name] = value
        except (ValueError, TypeError, AttributeError) as e:
            logging.error(
                "[DESCRIPTOR ERROR] Ошибка при вычислении '%s' для SMILES '%s': %s",
                name,
                smiles,
                str(e),
            )
            descriptors[name] = None

    return descriptors
