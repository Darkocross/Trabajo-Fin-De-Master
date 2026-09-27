# src/models/random_forest.py

# ============================================================
# RANDOM FOREST
# ============================================================
#
# Primer modelo no lineal del proyecto.
#
# Un Random Forest promedia muchos árboles de decisión
# entrenados sobre muestras distintas de los datos. Aporta dos
# cosas que la regresión lineal no puede dar:
#
#   1. Relaciones no lineales.
#      La demanda no crece de forma indefinida con la
#      temperatura: sube hasta un punto agradable y vuelve a
#      bajar con el calor extremo. Una recta no puede
#      representar esa forma; un árbol sí.
#
#   2. Interacciones automáticas.
#      No hace falta declarar "sábado de junio": el árbol
#      encuentra por sí mismo las combinaciones que importan.
#
# A cambio pierde interpretabilidad directa y no puede
# extrapolar fuera del rango de valores que ha visto.
#
# Los hiperparámetros se mantienen conservadores porque el
# conjunto de entrenamiento es pequeño (unos 700 días) y un
# bosque demasiado profundo memorizaría el ruido.
#
# ============================================================

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
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
    Construye el pipeline del Random Forest.

    No se estandarizan las variables numéricas: los árboles
    dividen por umbrales y son insensibles a la escala.
    """

    return Pipeline([
        (
            "preprocesamiento",
            crear_preprocesador(escalar=False),
        ),
        (
            "modelo",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=SEMILLA,
                n_jobs=-1,
            ),
        ),
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    return entrenar_y_evaluar(
        nombre_tecnico="random_forest",
        nombre_legible="Random Forest",
        modelo=crear_modelo(),
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
