# tests/conftest.py

# ============================================================
# CONFIGURACIÓN COMÚN DE LOS TESTS
# ============================================================
#
# Añade la raíz del proyecto al path para que los tests puedan
# importar `src` sin instalar el paquete, y define los datos
# que comparten varios tests.
#
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import CALENDARIO, GOLD_DATASET


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def calendario():
    """
    Calendario generado por el proyecto.
    """

    if not CALENDARIO.exists():

        pytest.skip(
            "No existe el calendario. Ejecuta: "
            "python pipeline.py --hasta calendario"
        )

    return pd.read_csv(CALENDARIO, parse_dates=["fecha"])


@pytest.fixture(scope="session")
def gold():
    """
    Capa gold del proyecto.
    """

    if not GOLD_DATASET.exists():

        pytest.skip(
            "No existe la capa gold. Ejecuta: "
            "python pipeline.py --hasta gold"
        )

    return pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])
