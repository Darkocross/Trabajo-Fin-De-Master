# src/models/regresion_lineal.py

# ============================================================
# REGRESIÓN LINEAL
# ============================================================
#
# Primer modelo de machine learning del proyecto.
#
# Se elige por tres motivos:
#
#   1. Es interpretable. Cada coeficiente dice cuántos
#      clientes más o menos aporta una variable, y eso se
#      puede explicar al responsable del restaurante.
#
#   2. Sirve para comprobar si la relación entre las variables
#      y la demanda es aproximadamente lineal. Si un modelo no
#      lineal la supera con claridad, sabremos que el problema
#      tiene estructura que la recta no captura.
#
#   3. Es rápida y estable, y no tiene apenas hiperparámetros
#      que ajustar.
#
# Las variables numéricas se estandarizan para que los
# coeficientes sean comparables entre sí y puedan leerse como
# una medida de importancia.
#
# ============================================================

import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.models.entrenamiento import entrenar_y_evaluar
from src.models.features import crear_preprocesador


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():
    """
    Construye el pipeline de la regresión lineal.

    El preprocesamiento va dentro del pipeline, no fuera: así
    la imputación y la codificación se aprenden solo con los
    datos de entrenamiento y se aplican igual en validación,
    en test y en producción.
    """

    return Pipeline([
        (
            "preprocesamiento",
            crear_preprocesador(escalar=True),
        ),
        (
            "modelo",
            LinearRegression(),
        ),
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    return entrenar_y_evaluar(
        nombre_tecnico="regresion_lineal",
        nombre_legible="Regresión lineal",
        modelo=crear_modelo(),
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
