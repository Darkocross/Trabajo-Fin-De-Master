# src/analysis/comparar_variables.py

# ============================================================
# COMPARACIÓN DE CONJUNTOS DE VARIABLES
# ============================================================
#
# Este módulo existe para responder a una pregunta concreta:
#
#   ¿Merecen la pena las interacciones y las variables de
#   calendario opcionales, o solo añaden columnas?
#
# ------------------------------------------------------------
# POR QUÉ HACE FALTA
# ------------------------------------------------------------
#
# Una versión anterior del proyecto respondía que sí, pero sin
# medirlo. El calendario publicaba `es_fin_semana_mayo`,
# `es_fin_semana_junio` y `es_fin_semana_julio`, y el
# argumento era razonable sobre el papel: "un sábado de junio
# no es la suma de sábado y junio".
#
# El problema no era el argumento, era el origen de los tres
# meses. No salieron de los datos: salieron de los tres meses
# que el generador sintético amplifica. Es fuga conceptual, y
# además una mala elección incluso aceptándola: la ratio
# findes/laborables más alta de la serie está en noviembre
# (1,955) y en abril (1,913), por encima de mayo (1,839).
#
# La forma honesta de plantearlo es la de este módulo: se
# construye la interacción para los DOCE meses, se compara
# contra el conjunto de efectos principales, y se decide con
# el resultado delante.
#
# ------------------------------------------------------------
# CÓMO SE COMPARA
# ------------------------------------------------------------
#
# Con tres medidas, porque una sola engaña:
#
#   1. MAE en validación (2025). La medida de decisión.
#   2. MAE en test (2026). Se mira una vez, al final.
#   3. MAE en validación cruzada temporal, con su desviación
#      típica entre bloques. Es la que dice si una diferencia
#      es real o cabe dentro del ruido.
#
# El criterio: una variante solo gana si mejora MÁS de una
# desviación típica. Si no, gana la más simple.
#
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import PROCESSED_RESULTADOS_DIR

from src.models.entrenamiento import (
    calcular_metricas,
    cargar_conjuntos,
)

from src.models.features import (
    VARIABLE_OBJETIVO,
    anadir_variables_derivadas,
)


# ============================================================
# PARÁMETROS DE LA VALIDACIÓN CRUZADA
# ============================================================
#
# Los mismos que usa src/models/validacion_cruzada.py, para
# que las cifras sean comparables entre módulos.

BLOQUES = 5

DIAS_POR_BLOQUE = 120


SALIDA = (
    PROCESSED_RESULTADOS_DIR / "comparacion_variables.csv"
)


# ============================================================
# LAS VARIANTES
# ============================================================
#
# Cada variante es un conjunto de variables. La primera es la
# que está desplegada; las demás añaden bloques encima.

CATEGORICAS_BASE = [
    "dia_semana",
    "mes",
    "tipo_vacaciones",
]

NUMERICAS_BASE = [
    "es_festivo",
    "tmed",
    "amplitud_termica",
    "prec_log",
    "lluvia_fin_semana",
]

CALENDARIO_OPCIONAL = [
    "es_vispera_festivo",
    "es_puente",
]


VARIANTES = [
    {
        "clave": "A",
        "nombre": "Solo efectos principales",
        "descripcion": (
            "El conjunto desplegado. Día de la semana, mes, "
            "vacaciones, festivo y meteorología."
        ),
        "categoricas": CATEGORICAS_BASE,
        "numericas": NUMERICAS_BASE,
    },
    {
        "clave": "B",
        "nombre": "+ interacción mes x fin de semana",
        "descripcion": (
            "Añade `mes_finde`: la interacción para los doce "
            "meses, no para tres elegidos a dedo."
        ),
        "categoricas": CATEGORICAS_BASE + ["mes_finde"],
        "numericas": NUMERICAS_BASE,
    },
    {
        "clave": "C",
        "nombre": "+ vísperas y puentes",
        "descripcion": (
            "Añade `es_vispera_festivo` y `es_puente`, dos "
            "hechos del almanaque que el calendario publica."
        ),
        "categoricas": CATEGORICAS_BASE,
        "numericas": NUMERICAS_BASE + CALENDARIO_OPCIONAL,
    },
    {
        "clave": "D",
        "nombre": "Todo encendido",
        "descripcion": (
            "Interacción y calendario opcional a la vez."
        ),
        "categoricas": CATEGORICAS_BASE + ["mes_finde"],
        "numericas": NUMERICAS_BASE + CALENDARIO_OPCIONAL,
    },
]


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

def preparar(df):
    """
    Deja el conjunto listo para cualquiera de las variantes.

    No se usa `preparar_dataset` de features.py a propósito:
    ese aplica los interruptores del módulo, y aquí hace falta
    tener todas las columnas disponibles a la vez para poder
    encenderlas y apagarlas variante a variante.
    """

    df = df.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    for columna in (
        "es_festivo",
        "es_vispera_festivo",
        "es_puente",
        "fin_de_semana",
        "tmax",
        "tmin",
        "tmed",
        "prec",
        VARIABLE_OBJETIVO,
    ):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    for columna in ("dia_semana", "mes", "tipo_vacaciones"):

        df[columna] = (
            df[columna]
            .astype("string")
            .fillna("desconocido")
            .astype(str)
        )

    df = anadir_variables_derivadas(df)

    df = df.dropna(subset=[VARIABLE_OBJETIVO])

    return df.sort_values("fecha").reset_index(drop=True)


def crear_modelo(categoricas, numericas):
    """
    Regresión lineal con el preprocesamiento del proyecto.

    Se usa la regresión y no el boosting porque es el modelo
    desplegado y el más sensible a que sobren columnas: si una
    variante aporta algo, aquí se nota.
    """

    return Pipeline([
        (
            "preprocesamiento",
            ColumnTransformer([
                (
                    "categoricas",
                    Pipeline([
                        (
                            "imputacion",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]),
                    categoricas,
                ),
                (
                    "numericas",
                    Pipeline([
                        (
                            "imputacion",
                            SimpleImputer(strategy="median"),
                        ),
                        ("escalado", StandardScaler()),
                    ]),
                    numericas,
                ),
            ]),
        ),
        ("modelo", LinearRegression()),
    ])


# ============================================================
# EVALUACIÓN DE UNA VARIANTE
# ============================================================

def evaluar_variante(variante, train, validacion, test):

    columnas = variante["categoricas"] + variante["numericas"]

    def xy(conjunto):
        return (
            conjunto[columnas],
            conjunto[VARIABLE_OBJETIVO],
        )

    X_train, y_train = xy(train)
    X_val, y_val = xy(validacion)
    X_test, y_test = xy(test)

    modelo = crear_modelo(
        variante["categoricas"],
        variante["numericas"],
    )

    modelo.fit(X_train, y_train)

    n_columnas = (
        modelo
        .named_steps["preprocesamiento"]
        .transform(X_train.head(1))
        .shape[1]
    )

    metricas_val = calcular_metricas(
        y_val,
        modelo.predict(X_val),
    )

    metricas_test = calcular_metricas(
        y_test,
        modelo.predict(X_test),
    )

    # --------------------------------------------------------
    # VALIDACIÓN CRUZADA TEMPORAL
    # --------------------------------------------------------
    #
    # Sobre train + validación, nunca sobre test.

    historico = pd.concat(
        [train, validacion],
        ignore_index=True,
    ).sort_values("fecha")

    X_hist, y_hist = xy(historico)

    X_hist = X_hist.reset_index(drop=True)
    y_hist = y_hist.reset_index(drop=True)

    separador = TimeSeriesSplit(
        n_splits=BLOQUES,
        test_size=DIAS_POR_BLOQUE,
    )

    maes = []

    for indices_train, indices_val in separador.split(X_hist):

        modelo_bloque = crear_modelo(
            variante["categoricas"],
            variante["numericas"],
        )

        modelo_bloque.fit(
            X_hist.iloc[indices_train],
            y_hist.iloc[indices_train],
        )

        maes.append(
            calcular_metricas(
                y_hist.iloc[indices_val],
                modelo_bloque.predict(
                    X_hist.iloc[indices_val]
                ),
            )["mae"]
        )

    return {
        "clave": variante["clave"],
        "variante": variante["nombre"],
        "columnas": int(n_columnas),
        "mae_validacion": round(metricas_val["mae"], 3),
        "mae_test": round(metricas_test["mae"], 3),
        "r2_test": round(metricas_test["r2"], 3),
        "mae_cruzada": round(float(np.mean(maes)), 3),
        "desviacion_cruzada": round(float(np.std(maes)), 3),
        "descripcion": variante["descripcion"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("COMPARACIÓN DE CONJUNTOS DE VARIABLES")
    print("=" * 60)

    train, validacion, test = cargar_conjuntos(verbose=False)

    train = preparar(train)
    validacion = preparar(validacion)
    test = preparar(test)

    filas = [
        evaluar_variante(variante, train, validacion, test)
        for variante in VARIANTES
    ]

    resultados = pd.DataFrame(filas)

    # --------------------------------------------------------
    # TABLA
    # --------------------------------------------------------

    print()
    print("RESULTADOS")
    print("-" * 60)

    print(
        resultados[[
            "clave",
            "variante",
            "columnas",
            "mae_validacion",
            "mae_test",
            "mae_cruzada",
            "desviacion_cruzada",
        ]].to_string(index=False)
    )

    # --------------------------------------------------------
    # LECTURA
    # --------------------------------------------------------

    base = resultados[resultados["clave"] == "A"].iloc[0]

    print()
    print("LECTURA")
    print("-" * 60)

    print(
        "Criterio: una variante solo gana si mejora el MAE de "
        "validación cruzada"
    )

    print(
        "en más de una desviación típica sobre los efectos "
        "principales."
    )

    print()

    umbral = base["mae_cruzada"] - base["desviacion_cruzada"]

    print(
        f"Referencia (A): {base['mae_cruzada']:.3f} "
        f"± {base['desviacion_cruzada']:.3f}   "
        f"-> hay que bajar de {umbral:.3f}"
    )

    print()

    ganadoras = []

    for _, fila in resultados.iterrows():

        if fila["clave"] == "A":
            continue

        gana = fila["mae_cruzada"] < umbral

        if gana:
            ganadoras.append(fila["clave"])

        diferencia = fila["mae_cruzada"] - base["mae_cruzada"]

        print(
            f"  {fila['clave']}  "
            f"{fila['mae_cruzada']:.3f}   "
            f"({diferencia:+.3f})   "
            f"+{fila['columnas'] - base['columnas']} columnas"
            f"   ->  {'GANA' if gana else 'dentro del ruido'}"
        )

    print()

    if ganadoras:

        print(
            "Variantes que superan el umbral: "
            f"{', '.join(ganadoras)}."
        )

        print(
            "Conviene encender los interruptores "
            "correspondientes en src/models/features.py."
        )

    else:

        print(
            "Ninguna variante supera el umbral. Todas las "
            "diferencias caben dentro del"
        )

        print(
            "ruido de la validación cruzada, así que gana la "
            "más simple: se despliega A,"
        )

        print(
            "y los interruptores de features.py se quedan "
            "apagados."
        )

        print()

        print(
            "Esto NO significa que la interacción no exista: "
            "el generador sí la produce"
        )

        print(
            "(entre un 5 % y un 8 % los fines de semana de "
            "mayo, junio y julio). Significa"
        )

        print(
            "que con ~1.100 observaciones ese efecto es más "
            "pequeño que el error de"
        )

        print(
            "estimación de las veintiuna columnas que hacen "
            "falta para capturarlo."
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
