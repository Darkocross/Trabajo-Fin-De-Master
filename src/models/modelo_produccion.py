# src/models/modelo_produccion.py

# ============================================================
# MODELO DE PRODUCCIÓN
# ============================================================
#
# Los cuatro modelos anteriores se entrenan con 2022-2024 y se
# comparan con la validación de 2025. Ese protocolo sirve para
# ELEGIR modelo de forma honesta, pero no es el que se despliega.
#
# Antes de poner un modelo a funcionar se vuelve a entrenar con
# TODOS los datos disponibles hasta la fecha (2022-2025), por
# una razón sencilla: cuanto más reciente sea el último dato
# que ha visto, mejor predice. Los hiperparámetros y el tipo de
# modelo ya están decididos, así que no hay riesgo de elegir
# mirando el test.
#
# El conjunto de 2026 sigue sin usarse para entrenar, de modo
# que las métricas que se publican sobre él siguen siendo una
# estimación limpia.
#
# Este es el modelo que carga la aplicación y el que usa la
# predicción diaria.
#
# ============================================================

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    COMPARACION_MODELOS,
    MODELOS_DIR,
    MODELO_PRODUCCION,
    ruta_modelo,
)

from src.models.entrenamiento import (
    calcular_intervalo,
    calcular_metricas,
    cargar_conjuntos,
    guardar_metadatos,
    mostrar_metricas,
)

from src.models.features import separar_x_y


# ============================================================
# CONSTRUCTORES DE MODELO
# ============================================================
#
# Se importan de forma perezosa para que este módulo no
# dependa de todos los modelos si solo se necesita uno.

def obtener_constructor(nombre_tecnico):
    """
    Devuelve la función que construye el modelo indicado.
    """

    if nombre_tecnico == "regresion_lineal":

        from src.models.regresion_lineal import crear_modelo

    elif nombre_tecnico == "random_forest":

        from src.models.random_forest import crear_modelo

    elif nombre_tecnico == "hist_gradient_boosting":

        from src.models.hist_gradient_boosting import (
            crear_modelo,
        )

    elif nombre_tecnico == "baseline":

        from src.models.baseline import crear_modelo

    else:

        raise ValueError(
            f"Modelo desconocido: {nombre_tecnico}"
        )

    return crear_modelo


# ============================================================
# SELECCIÓN DEL MODELO
# ============================================================

def seleccionar_modelo():
    """
    Devuelve el nombre técnico del modelo que se despliega.

    El criterio es el MAE en el conjunto de VALIDACIÓN, la
    métrica declarada como principal antes de ver ningún
    resultado. El conjunto de test no interviene en la
    elección: si interviniese, dejaría de ser una estimación
    imparcial del error.

    `MODELO_PRODUCCION` en `src/config.py` fija el modelo
    elegido para que el resultado sea reproducible. Si la
    comparación indica otro ganador, se avisa.
    """

    if not COMPARACION_MODELOS.exists():

        return MODELO_PRODUCCION

    comparacion = pd.read_csv(COMPARACION_MODELOS)

    if comparacion.empty:
        return MODELO_PRODUCCION

    ganador = (
        comparacion
        .sort_values("mae_validacion")
        .iloc[0]
    )

    if ganador["nombre_tecnico"] != MODELO_PRODUCCION:

        print()
        print("AVISO")
        print("-" * 60)

        print(
            "El mejor modelo en validación es "
            f"'{ganador['nombre_tecnico']}', pero en "
            f"src/config.py está fijado "
            f"'{MODELO_PRODUCCION}'."
        )

        print(
            "Revisa MODELO_PRODUCCION si quieres desplegar "
            "el ganador."
        )

    return MODELO_PRODUCCION


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO DE PRODUCCIÓN")
    print("=" * 60)

    # --------------------------------------------------------
    # CONJUNTOS
    # --------------------------------------------------------

    train, validacion, test = cargar_conjuntos()

    nombre_tecnico = seleccionar_modelo()

    print()
    print(f"Modelo seleccionado: {nombre_tecnico}")

    # --------------------------------------------------------
    # ENTRENAR CON TODO EL HISTÓRICO DISPONIBLE
    # --------------------------------------------------------

    historico = pd.concat(
        [train, validacion],
        ignore_index=True,
    ).sort_values("fecha").reset_index(drop=True)

    print()
    print(
        f"Entrenando con {len(historico)} registros "
        f"({historico['fecha'].min().date()} -> "
        f"{historico['fecha'].max().date()})"
    )

    crear_modelo = obtener_constructor(nombre_tecnico)

    modelo = crear_modelo()

    X_historico, y_historico = separar_x_y(historico)

    modelo.fit(X_historico, y_historico)

    # --------------------------------------------------------
    # EVALUACIÓN SOBRE 2026
    # --------------------------------------------------------
    #
    # 2026 no ha intervenido en el entrenamiento, así que estas
    # métricas siguen siendo una estimación limpia del error
    # que tendría el modelo desplegado.

    X_test, y_test = separar_x_y(test)

    pred_test = np.clip(
        modelo.predict(X_test),
        0,
        None,
    )

    metricas_test = calcular_metricas(y_test, pred_test)

    mostrar_metricas(
        "MODELO DE PRODUCCIÓN · TEST (2026)",
        metricas_test,
    )

    # --------------------------------------------------------
    # SESGO
    # --------------------------------------------------------

    sesgo = float(
        np.mean(
            np.asarray(y_test, dtype=float) - pred_test
        )
    )

    print()
    print(
        f"Sesgo medio (real - predicho): {sesgo:+.1f} clientes"
    )

    if sesgo > 5:

        print()
        print(
            "El modelo se queda corto de forma sistemática. "
            "Es el efecto del crecimiento anual del negocio, "
            "que un modelo de árboles no puede extrapolar. "
            "La solución operativa es reentrenar de forma "
            "periódica."
        )

    # --------------------------------------------------------
    # INTERVALO DE PREDICCIÓN
    # --------------------------------------------------------
    #
    # Se calcula sobre los residuos del último año completo
    # disponible en el entrenamiento (2025), que es el periodo
    # más parecido al que se va a predecir.

    ultimo_año = validacion.copy()

    X_ultimo, y_ultimo = separar_x_y(ultimo_año)

    pred_ultimo = np.clip(
        modelo.predict(X_ultimo),
        0,
        None,
    )

    residuos = (
        np.asarray(y_ultimo, dtype=float) - pred_ultimo
    )

    inferior, superior = calcular_intervalo(
        residuos,
        nivel=0.80,
    )

    print()
    print("INTERVALO DE PREDICCIÓN (80 %)")
    print("-" * 60)
    print(
        f"Predicción {inferior:+.0f} / {superior:+.0f} "
        "clientes"
    )

    print()
    print(
        "Nota: el intervalo se calcula sobre datos que el "
        "modelo de producción ya ha visto, así que es "
        "optimista. El intervalo honesto es el del modelo "
        "de selección, que aparece en la comparación."
    )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    ruta = ruta_modelo("produccion")

    joblib.dump(modelo, ruta, compress=3)

    guardar_metadatos(
        "produccion",
        {
            "nombre": "Modelo de producción",
            "algoritmo": nombre_tecnico,
            "registros_entrenamiento": int(len(historico)),
            "periodo_entrenamiento": [
                str(historico["fecha"].min().date()),
                str(historico["fecha"].max().date()),
            ],
            "metricas_test": metricas_test,
            "sesgo_test": sesgo,
            "intervalo_80": {
                "inferior": inferior,
                "superior": superior,
            },
        },
    )

    print()
    print("Modelo de producción guardado en:")
    print(f"  {ruta}")

    print()
    print("=" * 60)
    print("MODELO DE PRODUCCIÓN LISTO")
    print("=" * 60)

    return modelo, metricas_test


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
