import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Descriptors3D import CalcMolDescriptors3D

# Отключение RDKit-логов
RDLogger.DisableLog("rdApp.*")

# Лог-файл для ошибок дескрипторов
logging.basicConfig(
    filename="logs/descriptors3d_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Список имён 3D-дескрипторов
descriptor_3d_names = [
    "PMI1",
    "PMI2",
    "PMI3",
    "NPR1",
    "NPR2",
    "RadiusOfGyration",
    "InertialShapeFactor",
    "Eccentricity",
    "Asphericity",
    "SpherocityIndex",
    "PBF",
]


def compute_3d_descriptors(smiles: str) -> dict:
    """
    Вычисляет 3D-дескрипторы для молекулы, заданной SMILES-строкой.

    Args:
        smiles (str): SMILES-представление молекулы.

    Returns:
        dict: Словарь {имя_дескриптора: значение}, или None при ошибке.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning("Не удалось создать молекулу из SMILES: %s", smiles)
        return {name: None for name in descriptor_3d_names}

    mol = Chem.AddHs(mol)

    try:
        # Генерация 3D-конформера
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if success != 0:
            logging.error(
                "Не удалось сгенерировать 3D-конформер для SMILES: %s", smiles
            )
            return {name: None for name in descriptor_3d_names}

        # Оптимизация геометрии
        success = AllChem.UFFOptimizeMolecule(mol, maxIters=500, vdwThresh=10.0)
        if success != 0:
            logging.warning(
                "3D-оптимизация завершилась с ошибкой для SMILES: %s", smiles
            )

        # Вычисление дескрипторов
        descriptors_dict = CalcMolDescriptors3D(mol)
        return descriptors_dict

    except (ValueError, TypeError, RuntimeError) as e:
        logging.error(
            "Ошибка при расчёте 3D-дескрипторов для SMILES %s: %s", smiles, str(e)
        )
        return {name: None for name in descriptor_3d_names}
