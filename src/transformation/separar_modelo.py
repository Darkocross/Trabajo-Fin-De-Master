# src/transformation/separar_modelo.py

# ============================================================
# SEPARACIÓN TEMPORAL DEL DATASET
# ============================================================
#
# Divide la capa gold en los tres conjuntos de modelado.
#
# La separación es TEMPORAL, no aleatoria:
#
#     train      -> 2022, 2023 y 2024
#     validación -> 2025
#     test       -> 2026
#
# Una separación aleatoria mezclaría días de 2026 en el
# entrenamiento y días de 2022 en el test. El modelo estaría
# aprendiendo del futuro y la métrica resultante sería
# demasiado optimista: no se parecería en nada al error que
# tendría el sistema funcionando de verdad, donde solo se
# conoce el pasado.
#
# La regla de separación está definida una única vez, en
# `src/models/features.py`, y se usa igual en todo el
# proyecto.
#
# ============================================================

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    GOLD_DATASET,
    PROCESSED_MODELADO_DIR,
    TEST,
    TRAIN,
    VALIDATION,
)

from src.models.features import (
    separar_temporal,
    validar_separacion,
)


# ============================================================
# CARGAR LA CAPA GOLD
# ============================================================

def cargar_gold():
    """
    Carga la capa gold y la ordena cronológicamente.
    """

    print("Cargando la capa gold...")
    print(f"Archivo: {GOLD_DATASET}")

    if not GOLD_DATASET.exists():

        raise FileNotFoundError(
            f"No existe la capa gold: {GOLD_DATASET}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py --hasta gold"
        )

    gold = pd.read_csv(
        GOLD_DATASET,
        parse_dates=["fecha"],
    )

    if gold.empty:

        raise ValueError("La capa gold está vacía.")

    gold = (
        gold
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    print(f"Registros cargados: {len(gold)}")

    return gold


# ============================================================
# GUARDAR
# ============================================================

def guardar_conjuntos(train, validacion, test):
    """
    Guarda los tres conjuntos en data/processed/modelado.
    """

    PROCESSED_MODELADO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ruta, conjunto in [
        (TRAIN, train),
        (VALIDATION, validacion),
        (TEST, test),
    ]:

        conjunto.to_csv(
            ruta,
            index=False,
            encoding="utf-8",
        )

    print()
    print("Conjuntos guardados correctamente.")
    print(f"Train:      {TRAIN}")
    print(f"Validación: {VALIDATION}")
    print(f"Test:       {TEST}")


# ============================================================
# RESUMEN
# ============================================================

def mostrar_resumen(train, validacion, test):
    """
    Muestra el reparto de registros entre los conjuntos.
    """

    print()
    print("=" * 60)
    print("RESUMEN DE LA SEPARACIÓN")
    print("=" * 60)
    print()

    total = len(train) + len(validacion) + len(test)

    for nombre, conjunto in [
        ("TRAIN", train),
        ("VALIDACIÓN", validacion),
        ("TEST", test),
    ]:

        porcentaje = 100 * len(conjunto) / total

        print(f"{nombre}")

        print(
            f"  Registros: {len(conjunto)} "
            f"({porcentaje:.1f} %)"
        )

        print(
            f"  Periodo:   "
            f"{conjunto['fecha'].min().date()} -> "
            f"{conjunto['fecha'].max().date()}"
        )

        print(
            f"  Clientes:  "
            f"media {conjunto['n_clientes'].mean():.1f}, "
            f"máximo {conjunto['n_clientes'].max()}"
        )

        print()

    print(f"Total registros: {total}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SEPARACIÓN TEMPORAL DEL DATASET")
    print("=" * 60)
    print()

    gold = cargar_gold()

    train, validacion, test = separar_temporal(gold)

    validar_separacion(train, validacion, test)

    print()
    print("Los conjuntos son temporalmente consistentes.")

    guardar_conjuntos(train, validacion, test)

    mostrar_resumen(train, validacion, test)

    print()
    print("=" * 60)
    print("SEPARACIÓN FINALIZADA CORRECTAMENTE")
    print("=" * 60)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
