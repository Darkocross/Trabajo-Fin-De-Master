# src/models/validacion_cruzada.py

# ============================================================
# VALIDACIÓN CRUZADA TEMPORAL
# ============================================================
#
# Comparar modelos con una sola partición tiene un problema:
# la diferencia entre el primero y el segundo puede ser más
# pequeña que el ruido de esa partición concreta. Elegir así
# equivale a elegir por casualidad.
#
# Esta comprobación repite la comparación cinco veces, sobre
# cinco periodos distintos, siempre entrenando con el pasado y
# evaluando con el futuro inmediato (origen deslizante):
#
#     entrena [-------]  evalúa [---]
#     entrena [-----------]  evalúa [---]
#     entrena [---------------]  evalúa [---]
#     ...
#
# Es exactamente lo que ocurriría en la realidad: cada vez que
# se reentrena el modelo, se hace con todo lo que ha pasado
# hasta ese momento y se usa para predecir lo que viene.
#
# Con cinco medidas por modelo ya se puede saber si una
# diferencia es real o es ruido.
#
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import PROCESSED_RESULTADOS_DIR

from src.models.baseline import crear_modelo as crear_baseline

from src.models.entrenamiento import (
    calcular_metricas,
    cargar_conjuntos,
)

from src.models.features import separar_x_y

from src.models.hist_gradient_boosting import (
    crear_modelo as crear_boosting,
)

from src.models.random_forest import (
    crear_modelo as crear_random_forest,
)

from src.models.regresion_lineal import (
    crear_modelo as crear_lineal,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELOS = [
    ("baseline", "Baseline", crear_baseline),
    ("regresion_lineal", "Regresión lineal", crear_lineal),
    ("random_forest", "Random Forest", crear_random_forest),
    (
        "hist_gradient_boosting",
        "Hist Gradient Boosting",
        crear_boosting,
    ),
]


# Número de bloques de evaluación.
BLOQUES = 5

# Días de cada bloque de evaluación (unos cuatro meses de
# actividad, que cubre estaciones distintas).
DIAS_POR_BLOQUE = 120


SALIDA = (
    PROCESSED_RESULTADOS_DIR / "validacion_cruzada.csv"
)


# ============================================================
# VALIDACIÓN CRUZADA
# ============================================================

def validar_modelo(crear_modelo, X, y, particiones):
    """
    Entrena y evalúa un modelo en cada bloque.
    """

    errores = []

    for indices_train, indices_test in particiones:

        modelo = crear_modelo()

        modelo.fit(
            X.iloc[indices_train],
            y.iloc[indices_train],
        )

        predicciones = np.clip(
            modelo.predict(X.iloc[indices_test]),
            0,
            None,
        )

        metricas = calcular_metricas(
            y.iloc[indices_test],
            predicciones,
        )

        errores.append(metricas["mae"])

    return errores


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VALIDACIÓN CRUZADA TEMPORAL")
    print("=" * 60)

    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------
    #
    # Se usa todo el histórico disponible salvo el test, que se
    # reserva como estimación final.

    train, validacion, test = cargar_conjuntos(verbose=False)

    historico = (
        pd.concat([train, validacion], ignore_index=True)
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    X, y = separar_x_y(historico)

    print()
    print(
        f"Datos: {len(historico)} días "
        f"({historico['fecha'].min().date()} -> "
        f"{historico['fecha'].max().date()})"
    )

    print(
        f"Bloques: {BLOQUES} de {DIAS_POR_BLOQUE} días, "
        "con origen deslizante"
    )

    separador = TimeSeriesSplit(
        n_splits=BLOQUES,
        test_size=DIAS_POR_BLOQUE,
    )

    particiones = list(separador.split(X))

    # --------------------------------------------------------
    # PERIODOS DE CADA BLOQUE
    # --------------------------------------------------------

    print()
    print("BLOQUES DE EVALUACIÓN")
    print("-" * 60)

    for numero, (indices_train, indices_test) in enumerate(
        particiones,
        start=1,
    ):

        fechas_test = historico.iloc[indices_test]["fecha"]

        print(
            f"  Bloque {numero}: entrena con "
            f"{len(indices_train):>3} días, evalúa "
            f"{fechas_test.min().date()} -> "
            f"{fechas_test.max().date()}"
        )

    # --------------------------------------------------------
    # EVALUAR
    # --------------------------------------------------------

    print()
    print("RESULTADOS (error medio absoluto, en clientes)")
    print("-" * 60)
    print()

    resultados = {}

    filas = []

    for nombre_tecnico, nombre_legible, crear in MODELOS:

        errores = validar_modelo(crear, X, y, particiones)

        resultados[nombre_tecnico] = np.array(errores)

        media = float(np.mean(errores))
        desviacion = float(np.std(errores))

        print(
            f"  {nombre_legible:<24} "
            f"{media:6.2f} ± {desviacion:.2f}   "
            f"{[round(error, 1) for error in errores]}"
        )

        filas.append({
            "modelo": nombre_legible,
            "nombre_tecnico": nombre_tecnico,
            "mae_medio": round(media, 3),
            "mae_desviacion": round(desviacion, 3),
            **{
                f"bloque_{numero}": round(error, 3)
                for numero, error in enumerate(
                    errores,
                    start=1,
                )
            },
        })

    # --------------------------------------------------------
    # COMPARACIÓN ENTRE LOS DOS MEJORES
    # --------------------------------------------------------

    ranking = sorted(
        resultados.items(),
        key=lambda elemento: elemento[1].mean(),
    )

    mejor, segundo = ranking[0], ranking[1]

    diferencias = segundo[1] - mejor[1]

    victorias = int((diferencias > 0).sum())

    print()
    print("=" * 60)
    print("¿ES REAL LA DIFERENCIA?")
    print("=" * 60)

    nombres = {
        clave: legible
        for clave, legible, _ in MODELOS
    }

    print()

    print(
        f"Mejor modelo:   {nombres[mejor[0]]} "
        f"({mejor[1].mean():.2f})"
    )

    print(
        f"Segundo:        {nombres[segundo[0]]} "
        f"({segundo[1].mean():.2f})"
    )

    print()

    print(
        f"Diferencia media: {diferencias.mean():.2f} clientes "
        f"(desviación {diferencias.std():.2f})"
    )

    print(
        f"El mejor gana en {victorias} de "
        f"{len(diferencias)} bloques."
    )

    print()

    if victorias >= 4:

        print(
            f"La ventaja de {nombres[mejor[0]]} es "
            "consistente: gana en casi todos los periodos, no "
            "solo en la partición elegida. La diferencia es "
            "real."
        )

    elif victorias >= 3:

        print(
            "La ventaja es leve y no se mantiene en todos los "
            "periodos. Con una diferencia así, conviene "
            "quedarse con el modelo más simple e "
            "interpretable."
        )

    else:

        print(
            "Los modelos son indistinguibles. La elección "
            "debe basarse en interpretabilidad y coste, no en "
            "la métrica."
        )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(filas).to_csv(
        SALIDA,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Guardado en: {SALIDA}")

    print()
    print("=" * 60)
    print("VALIDACIÓN CRUZADA COMPLETADA")
    print("=" * 60)

    return pd.DataFrame(filas)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
