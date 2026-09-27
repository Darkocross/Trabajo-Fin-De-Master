# src/models/hist_gradient_boosting.py

# ============================================================
# HIST GRADIENT BOOSTING REGRESSOR
# ============================================================
#
# Modelo de boosting por histogramas.
#
# A diferencia del Random Forest, que entrena muchos árboles
# en paralelo y promedia, el boosting entrena árboles en
# cadena: cada uno se especializa en corregir el error que ha
# dejado el anterior.
#
# Suele ser el modelo más preciso en problemas tabulares como
# este, y funciona bien con conjuntos pequeños porque cada
# árbol es poco profundo.
#
# Los hiperparámetros están elegidos para un dataset de unos
# 700 días de entrenamiento:
#
#   - `learning_rate` bajo y muchas iteraciones: aprende
#     despacio, lo que reduce el sobreajuste.
#   - `max_leaf_nodes` pequeño: árboles simples.
#   - `early_stopping` con validación interna: deja de
#     entrenar cuando ya no mejora, en lugar de agotar
#     siempre todas las iteraciones.
#
# ============================================================

import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import SEMILLA
from src.models.entrenamiento import entrenar_y_evaluar
from src.models.features import crear_preprocesador


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():
    """
    Construye el pipeline del HistGradientBoostingRegressor.
    """

    return Pipeline([
        (
            "preprocesamiento",
            crear_preprocesador(escalar=False),
        ),
        (
            "modelo",
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=600,
                max_leaf_nodes=16,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=SEMILLA,
            ),
        ),
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    return entrenar_y_evaluar(
        nombre_tecnico="hist_gradient_boosting",
        nombre_legible="Hist Gradient Boosting",
        modelo=crear_modelo(),
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
