# src/models/baseline.py

# ============================================================
# MODELO BASELINE
# ============================================================
#
# El baseline es la referencia contra la que se mide todo lo
# demás. No es un modelo de machine learning: reproduce lo que
# haría un encargado con experiencia y sin ninguna herramienta.
#
#     "Los sábados suelen venir unos 150 clientes."
#
# Es decir: predice, para cada día, la media histórica de
# clientes de ese día de la semana.
#
# Si un modelo de machine learning no mejora claramente este
# baseline, no aporta valor y no merece la pena ponerlo en
# producción, por muy sofisticado que sea.
#
# Se implementa como un estimador compatible con scikit-learn
# para que pueda pasar exactamente por el mismo protocolo de
# entrenamiento y evaluación que el resto de modelos.
#
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.models.entrenamiento import entrenar_y_evaluar


# ============================================================
# ESTIMADOR
# ============================================================

class MediaPorDiaSemana(BaseEstimator, RegressorMixin):
    """
    Predice la media histórica de clientes del día de la
    semana correspondiente.

    Si en predicción aparece un día de la semana que no estaba
    en el entrenamiento, devuelve la media global.
    """

    def __init__(self, columna="dia_semana"):

        self.columna = columna

    def fit(self, X, y):

        y = pd.Series(
            np.asarray(y, dtype=float)
        )

        dias = (
            pd.Series(X[self.columna].values)
            .astype(str)
            .reset_index(drop=True)
        )

        self.medias_ = (
            y.groupby(dias).mean().to_dict()
        )

        self.media_global_ = float(y.mean())

        return self

    def predict(self, X):

        dias = (
            pd.Series(X[self.columna].values)
            .astype(str)
        )

        return (
            dias
            .map(self.medias_)
            .fillna(self.media_global_)
            .to_numpy(dtype=float)
        )


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():
    """
    Construye el baseline.

    No lleva preprocesamiento: trabaja directamente sobre la
    columna `dia_semana`, igual que haría una persona mirando
    el calendario.
    """

    return MediaPorDiaSemana(columna="dia_semana")


# ============================================================
# MAIN
# ============================================================

def main():

    modelo, metricas = entrenar_y_evaluar(
        nombre_tecnico="baseline",
        nombre_legible="Baseline (media por día de la semana)",
        modelo=crear_modelo(),
    )

    # --------------------------------------------------------
    # MEDIAS APRENDIDAS
    # --------------------------------------------------------

    nombres_dias = {
        "0": "lunes",
        "1": "martes",
        "2": "miércoles",
        "3": "jueves",
        "4": "viernes",
        "5": "sábado",
        "6": "domingo",
    }

    print()
    print("MEDIA HISTÓRICA APRENDIDA (2022-2024)")
    print("-" * 60)

    for clave in sorted(modelo.medias_, key=lambda x: str(x)):

        nombre = nombres_dias.get(str(clave), str(clave))

        print(
            f"  {nombre:<12} "
            f"{modelo.medias_[clave]:6.1f} clientes"
        )

    return modelo, metricas


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
