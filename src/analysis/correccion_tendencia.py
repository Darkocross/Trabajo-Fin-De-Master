# src/analysis/correccion_tendencia.py

# ============================================================
# CORRECCIÓN DEL SESGO POR CRECIMIENTO
# ============================================================
#
# El modelo desplegado se queda corto de forma sistemática: en
# el conjunto de test predice de media unos 10 clientes menos
# de los que hubo. No es ruido, es sesgo.
#
# La causa está identificada desde el análisis de errores: el
# negocio crece alrededor de un 4 % anual y NINGUNA variable
# del modelo le dice que ha pasado el tiempo. Un sábado de
# junio de 2026 le parece idéntico a uno de junio de 2022.
#
# La memoria lo declaraba como "la limitación más relevante y
# la que primero habría que atacar", y luego no se atacaba.
# Este módulo la ataca, con el mismo método que se usó para
# las interacciones (src/analysis/comparar_variables.py):
# se plantean las opciones, se miden todas con el mismo
# protocolo, y se decide con el resultado delante.
#
# ------------------------------------------------------------
# LAS CUATRO OPCIONES
# ------------------------------------------------------------
#
# A · SIN CORRECCIÓN
#     El modelo actual. Es la referencia.
#
# B · ÍNDICE TEMPORAL COMO VARIABLE
#     Se añade `indice_temporal` (días desde el inicio de la
#     serie) al conjunto de entrada.
#
#     Es la opción evidente y la que más fácil se rompe. Un
#     modelo lineal puede extrapolar una recta, así que
#     funcionará mientras el crecimiento siga siendo lineal, y
#     se equivocará en la dirección contraria en cuanto el
#     negocio se estanque. Los modelos de árboles ni siquiera
#     pueden extrapolar: fuera del rango visto, predicen el
#     último valor conocido.
#
# C · VENTANA MÓVIL
#     Entrenar solo con los últimos N días en lugar de con
#     todo el histórico. No modela el crecimiento: lo esquiva,
#     manteniendo el modelo siempre cerca del nivel actual.
#
#     Cuesta datos, que es justo lo que no sobra.
#
# D · CORRECCIÓN DE NIVEL
#     Dejar el modelo como está y sumarle el sesgo medio de
#     los últimos N días. Separa dos problemas distintos —la
#     forma de la demanda y su nivel— y solo corrige el
#     segundo.
#
#     Es lo que hace cualquier encargado con experiencia:
#     "el modelo se queda corto, súmale diez".
#
# ------------------------------------------------------------
# CÓMO SE MIDE
# ------------------------------------------------------------
#
# Con TRES medidas, porque aquí las dos primeras engañan:
#
#   - MAE. Cuánto se equivoca.
#   - SESGO MEDIO (real - predicho). Hacia dónde se equivoca.
#   - CALIDAD DE LA DECISIÓN. Qué plantilla sale de ahí.
#
# La tercera es la que manda en este proyecto, y aquí se ve por
# qué. Una opción puede bajar el MAE, arreglar el sesgo y aun
# así producir PEORES decisiones, si lo que hace es cambiar un
# error sistemático por otro de signo contrario.
#
# ------------------------------------------------------------
# Y CON DOS HORIZONTES
# ------------------------------------------------------------
#
# Extrapolar una tendencia no cuesta lo mismo a tres meses que
# a dos años, así que medir con un solo horizonte da una
# respuesta que parece sólida y no lo es:
#
#   - HORIZONTE CORTO. Entrenar con 2022-2025 y predecir 2026,
#     que es lo que hace el modelo de producción.
#
#   - HORIZONTE LARGO. Entrenar con 2022-2024 y predecir 2026,
#     que es lo que hace el protocolo de comparación, y lo que
#     le pasaría a un modelo que lleva año y medio sin
#     reentrenar.
#
# La diferencia entre ambos es, de hecho, el resultado más
# interesante de este módulo.
#
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import PROCESSED_RESULTADOS_DIR

from src.analysis.evaluacion_decision import evaluar_metodo

from src.models.entrenamiento import (
    calcular_metricas,
    cargar_conjuntos,
)

from src.models.features import (
    VARIABLE_OBJETIVO,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
    crear_preprocesador,
    preparar_dataset,
)

from src.models.regresion_lineal import crear_modelo


SALIDA = (
    PROCESSED_RESULTADOS_DIR / "correccion_tendencia.csv"
)


# Días de histórico para la ventana móvil y para estimar el
# nivel en la corrección. Dos años y dos meses, que es lo que
# un negocio real tendría a mano sin esfuerzo.

DIAS_VENTANA = 730

DIAS_NIVEL = 60


# ============================================================
# UTILIDADES
# ============================================================

def separar(conjunto):
    """
    Separa variables de entrada y objetivo.
    """

    return (
        conjunto[VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS],
        conjunto[VARIABLE_OBJETIVO],
    )


def medir(nombre, descripcion, y_real, y_pred):
    """
    Error, dirección del error y calidad de la decisión.
    """

    metricas = calcular_metricas(y_real, y_pred)

    sesgo = float(
        np.mean(np.asarray(y_real) - np.asarray(y_pred))
    )

    decision = evaluar_metodo(
        np.asarray(y_real, dtype=float),
        np.asarray(y_pred, dtype=float),
    )

    dias = len(y_real)

    return {
        "opcion": nombre,
        "descripcion": descripcion,
        "mae": round(metricas["mae"], 3),
        "r2": round(metricas["r2"], 3),
        "sesgo_medio": round(sesgo, 2),
        "sesgo_absoluto": round(abs(sesgo), 2),
        "pct_correctos": round(
            100 * float(decision["correcto"].sum()) / dias, 1
        ),
        "dias_falta": int(decision["infradotado"].sum()),
        "dias_exceso": int(decision["sobredotado"].sum()),
        "coste": round(
            float(decision["coste_total"].sum())
        ),
    }


# ============================================================
# LAS CUATRO OPCIONES
# ============================================================

def opcion_a(train, test):
    """
    Sin corrección. El modelo actual.
    """

    X_train, y_train = separar(train)
    X_test, y_test = separar(test)

    modelo = crear_modelo()
    modelo.fit(X_train, y_train)

    return medir(
        "A · Sin corrección",
        "El modelo desplegado. Referencia.",
        y_test,
        modelo.predict(X_test),
    )


def opcion_b(train, test):
    """
    `indice_temporal` como variable de entrada.
    """

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression

    numericas = VARIABLES_NUMERICAS + ["indice_temporal"]

    columnas = VARIABLES_CATEGORICAS + numericas

    preprocesador = crear_preprocesador(escalar=True)

    # Se reemplaza el bloque numérico para incluir el índice.
    preprocesador.transformers[1] = (
        preprocesador.transformers[1][0],
        preprocesador.transformers[1][1],
        numericas,
    )

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("modelo", LinearRegression()),
    ])

    modelo.fit(train[columnas], train[VARIABLE_OBJETIVO])

    return medir(
        "B · Índice temporal",
        "Añade los días transcurridos como variable.",
        test[VARIABLE_OBJETIVO],
        modelo.predict(test[columnas]),
    )


def opcion_c(train, test, dias=DIAS_VENTANA):
    """
    Ventana móvil: entrenar solo con el pasado reciente.
    """

    corte = train["fecha"].max() - pd.Timedelta(days=dias)

    reciente = train[train["fecha"] > corte]

    X_train, y_train = separar(reciente)
    X_test, y_test = separar(test)

    modelo = crear_modelo()
    modelo.fit(X_train, y_train)

    return medir(
        f"C · Ventana móvil ({dias} días)",
        f"Entrena solo con los últimos {dias} días "
        f"({len(reciente)} registros).",
        y_test,
        modelo.predict(X_test),
    )


def opcion_d(train, test, dias=DIAS_NIVEL):
    """
    Corrección de nivel a partir del sesgo reciente.

    El ajuste se estima SOLO con datos anteriores al periodo
    de test. Usar el sesgo del propio test sería mirar la
    respuesta antes de responder.
    """

    X_train, y_train = separar(train)
    X_test, y_test = separar(test)

    modelo = crear_modelo()
    modelo.fit(X_train, y_train)

    # --- Estimar el nivel con la cola del entrenamiento ---

    corte = train["fecha"].max() - pd.Timedelta(days=dias)

    cola = train[train["fecha"] > corte]

    X_cola, y_cola = separar(cola)

    ajuste = float(
        np.mean(y_cola - modelo.predict(X_cola))
    )

    prediccion = modelo.predict(X_test) + ajuste

    return medir(
        f"D · Corrección de nivel (+{ajuste:.1f})",
        f"Suma el sesgo medio de los últimos {dias} días "
        f"de entrenamiento.",
        y_test,
        prediccion,
    )


# ============================================================
# MAIN
# ============================================================

def evaluar_horizonte(etiqueta, entrenamiento, evaluacion):
    """
    Aplica las cuatro opciones a un escenario concreto.
    """

    filas = [
        opcion_a(entrenamiento, evaluacion),
        opcion_b(entrenamiento, evaluacion),
        opcion_c(entrenamiento, evaluacion),
        opcion_d(entrenamiento, evaluacion),
    ]

    for fila in filas:
        fila["horizonte"] = etiqueta

    return filas


def mostrar(resultados, etiqueta):

    tabla = resultados[resultados["horizonte"] == etiqueta]

    print()
    print(etiqueta.upper())
    print("-" * 60)

    print(
        tabla[[
            "opcion",
            "mae",
            "sesgo_medio",
            "pct_correctos",
            "dias_falta",
            "dias_exceso",
        ]].to_string(index=False)
    )


def main():

    print("=" * 60)
    print("CORRECCIÓN DEL SESGO POR CRECIMIENTO")
    print("=" * 60)

    train, validacion, test = cargar_conjuntos(verbose=False)

    train = preparar_dataset(train)
    validacion = preparar_dataset(validacion)
    test = preparar_dataset(test)

    historico = pd.concat(
        [train, validacion],
        ignore_index=True,
    ).sort_values("fecha")

    filas = []

    filas += evaluar_horizonte(
        "Horizonte corto (entrena hasta 2025, predice 2026)",
        historico,
        test,
    )

    filas += evaluar_horizonte(
        "Horizonte largo (entrena hasta 2024, predice 2026)",
        train,
        test,
    )

    resultados = pd.DataFrame(filas)

    for etiqueta in resultados["horizonte"].unique():
        mostrar(resultados, etiqueta)

    print()
    print(
        "Sesgo medio = real - predicho. Positivo, el modelo se "
        "queda corto;"
    )
    print("negativo, se pasa.")

    # --------------------------------------------------------
    # LECTURA
    # --------------------------------------------------------

    print()
    print("LECTURA")
    print("-" * 60)

    for etiqueta in resultados["horizonte"].unique():

        tabla = resultados[
            resultados["horizonte"] == etiqueta
        ].set_index("opcion")

        base = tabla.iloc[0]

        mejor = tabla["pct_correctos"].idxmax()

        print()
        print(etiqueta)

        print(
            f"  Mejor decisión: {mejor} "
            f"({tabla.loc[mejor, 'pct_correctos']:.1f} % "
            f"frente a {base['pct_correctos']:.1f} %)"
        )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    resultados.to_csv(SALIDA, index=False, encoding="utf-8")

    print()
    print(f"Guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
