# src/models/analisis_importancia.py

# ============================================================
# ANÁLISIS DE IMPORTANCIA DE LAS VARIABLES
# ============================================================
#
# Responde a la pregunta que de verdad le interesa al
# restaurante: ¿qué es lo que más condiciona cuánta gente
# viene?
#
# El análisis NO vuelve a entrenar los modelos: carga los que
# ya están guardados. Antes cada script entrenaba su propia
# copia con variables distintas, de modo que las importancias
# publicadas no correspondían a los modelos evaluados.
#
# Se usan dos medidas complementarias:
#
# 1. IMPORTANCIA POR PERMUTACIÓN (medida principal)
#
#    Se desordena al azar una variable y se mide cuánto empeora
#    el error. Si al romperla el modelo apenas empeora, esa
#    variable no le estaba aportando nada.
#
#    Es la única medida comparable entre modelos distintos,
#    porque no depende de cómo funciona el algoritmo por
#    dentro, sino de cuánto necesita esa variable.
#
#    Se calcula sobre VALIDACIÓN, no sobre entrenamiento: lo
#    que interesa es qué variables ayudan a predecir días que
#    el modelo no ha visto.
#
# 2. IMPORTANCIA NATIVA (medida secundaria)
#
#    - Regresión lineal: el coeficiente. Como las variables
#      numéricas están estandarizadas, los coeficientes son
#      comparables entre sí y además tienen SIGNO, que dice si
#      la variable sube o baja la demanda.
#
#    - Random Forest: la reducción de impureza.
#
#    - HistGradientBoosting: no expone importancia nativa, así
#      que solo tiene la de permutación.
#
# ============================================================

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    PROCESSED_GRAFICOS_DIR,
    PROCESSED_RESULTADOS_DIR,
    SEMILLA,
    ruta_grafico,
    ruta_importancia,
    ruta_modelo,
)

from src.models.entrenamiento import cargar_conjuntos, cargar_modelo

from src.models.features import (
    obtener_nombres_variables,
    separar_x_y,
    traducir_variable,
)


# ============================================================
# MODELOS A ANALIZAR
# ============================================================

MODELOS = [
    ("regresion_lineal", "Regresión lineal"),
    ("random_forest", "Random Forest"),
    ("hist_gradient_boosting", "Hist Gradient Boosting"),
]


# Número de repeticiones de la permutación. Cuantas más, más
# estable es la estimación y más tarda.
REPETICIONES = 30


# ============================================================
# IMPORTANCIA POR PERMUTACIÓN
# ============================================================

def importancia_permutacion(modelo, X, y):
    """
    Calcula la importancia por permutación sobre las variables
    originales, no sobre las columnas que salen del One-Hot.

    Esto es lo que interesa: saber cuánto aporta "el día de la
    semana", no cuánto aporta "que sea martes".
    """

    resultado = permutation_importance(
        modelo,
        X,
        y,
        n_repeats=REPETICIONES,
        random_state=SEMILLA,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    importancia = pd.DataFrame({
        "variable": X.columns,
        "importancia": resultado.importances_mean,
        "desviacion": resultado.importances_std,
    })

    importancia["variable_legible"] = (
        importancia["variable"].apply(traducir_variable)
    )

    return (
        importancia
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# IMPORTANCIA NATIVA
# ============================================================

def importancia_nativa(modelo, nombre_tecnico):
    """
    Devuelve la importancia interna del modelo, cuando existe.

    Se expresa sobre las columnas que produce el
    preprocesamiento, porque es ahí donde el modelo trabaja.
    """

    estimador = modelo.named_steps["modelo"]

    nombres = obtener_nombres_variables(modelo)

    # --- Regresión lineal: coeficientes con signo ---

    if hasattr(estimador, "coef_"):

        coeficientes = np.ravel(estimador.coef_)

        tabla = pd.DataFrame({
            "variable": nombres,
            "coeficiente": coeficientes,
            "importancia": np.abs(coeficientes),
        })

    # --- Random Forest: reducción de impureza ---

    elif hasattr(estimador, "feature_importances_"):

        tabla = pd.DataFrame({
            "variable": nombres,
            "coeficiente": np.nan,
            "importancia": estimador.feature_importances_,
        })

    else:

        return None

    tabla["variable_legible"] = (
        tabla["variable"].apply(traducir_variable)
    )

    return (
        tabla
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# GRÁFICO
# ============================================================

def crear_grafico(importancia, titulo, ruta, n=12):
    """
    Dibuja un gráfico de barras horizontales con las variables
    más importantes.
    """

    datos = importancia.head(n).iloc[::-1]

    altura = max(4.0, 0.42 * len(datos) + 1.6)

    figura, ejes = plt.subplots(figsize=(10, altura))

    ejes.barh(
        datos["variable_legible"],
        datos["importancia"],
        color="#2f6f9f",
        edgecolor="white",
    )

    if "desviacion" in datos.columns:

        ejes.errorbar(
            datos["importancia"],
            range(len(datos)),
            xerr=datos["desviacion"],
            fmt="none",
            ecolor="#1b4664",
            capsize=3,
            linewidth=1,
        )

    ejes.set_title(titulo, fontsize=13, pad=14)

    ejes.set_xlabel(
        "Aumento del error medio al desordenar la variable "
        "(clientes)"
    )

    ejes.grid(axis="x", alpha=0.3)

    ejes.spines["top"].set_visible(False)
    ejes.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


def crear_grafico_comparativo(importancias, ruta):
    """
    Compara la importancia por permutación de los tres modelos
    sobre las mismas variables.

    Sirve para ver si los modelos coinciden en qué es lo que
    manda. Cuando coinciden, la conclusión es mucho más sólida
    que si dependiera de un solo algoritmo.
    """

    tabla = None

    for nombre_legible, importancia in importancias.items():

        columna = (
            importancia
            .set_index("variable_legible")["importancia"]
            .rename(nombre_legible)
        )

        tabla = (
            columna.to_frame()
            if tabla is None
            else tabla.join(columna, how="outer")
        )

    tabla = tabla.fillna(0)

    tabla = tabla.loc[
        tabla.mean(axis=1).sort_values().index
    ].tail(12)

    figura, ejes = plt.subplots(figsize=(11, 7))

    posiciones = np.arange(len(tabla))

    alto = 0.26

    colores = ["#2f6f9f", "#5aa469", "#d98d3a"]

    for indice, columna in enumerate(tabla.columns):

        ejes.barh(
            posiciones + indice * alto,
            tabla[columna],
            height=alto,
            label=columna,
            color=colores[indice % len(colores)],
            edgecolor="white",
        )

    ejes.set_yticks(posiciones + alto)
    ejes.set_yticklabels(tabla.index)

    ejes.set_xlabel(
        "Aumento del error medio al desordenar la variable "
        "(clientes)"
    )

    ejes.set_title(
        "Importancia de las variables según cada modelo",
        fontsize=13,
        pad=14,
    )

    ejes.legend(loc="lower right")

    ejes.grid(axis="x", alpha=0.3)

    ejes.spines["top"].set_visible(False)
    ejes.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ANÁLISIS DE IMPORTANCIA DE LAS VARIABLES")
    print("=" * 60)

    train, validacion, test = cargar_conjuntos()

    X_validacion, y_validacion = separar_x_y(validacion)

    PROCESSED_RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

    importancias = {}

    for nombre_tecnico, nombre_legible in MODELOS:

        if not ruta_modelo(nombre_tecnico).exists():

            print()
            print(
                f"Modelo '{nombre_tecnico}' no entrenado. "
                "Se omite."
            )

            continue

        print()
        print("=" * 60)
        print(nombre_legible.upper())
        print("=" * 60)

        modelo = cargar_modelo(nombre_tecnico)

        # ----------------------------------------------------
        # PERMUTACIÓN
        # ----------------------------------------------------

        print()
        print(
            "Calculando importancia por permutación "
            f"({REPETICIONES} repeticiones)..."
        )

        permutacion = importancia_permutacion(
            modelo,
            X_validacion,
            y_validacion,
        )

        importancias[nombre_legible] = permutacion

        print()
        print("Variables más influyentes:")
        print()

        for _, fila in permutacion.head(8).iterrows():

            print(
                f"  {fila['variable_legible']:<32} "
                f"{fila['importancia']:6.2f} "
                f"± {fila['desviacion']:.2f} clientes"
            )

        # ----------------------------------------------------
        # IMPORTANCIA NATIVA
        # ----------------------------------------------------

        nativa = importancia_nativa(modelo, nombre_tecnico)

        if nativa is not None:

            ruta_nativa = (
                PROCESSED_RESULTADOS_DIR
                / f"coeficientes_{nombre_tecnico}.csv"
            )

            nativa.to_csv(
                ruta_nativa,
                index=False,
                encoding="utf-8",
            )

            if nativa["coeficiente"].notna().any():

                print()
                print(
                    "Coeficientes con mayor efecto "
                    "(en clientes):"
                )
                print()

                for _, fila in nativa.head(8).iterrows():

                    print(
                        f"  {fila['variable_legible']:<32} "
                        f"{fila['coeficiente']:+7.1f}"
                    )

        # ----------------------------------------------------
        # GUARDAR
        # ----------------------------------------------------

        permutacion.to_csv(
            ruta_importancia(nombre_tecnico),
            index=False,
            encoding="utf-8",
        )

        crear_grafico(
            permutacion,
            f"Importancia de las variables · {nombre_legible}",
            ruta_grafico(f"importancia_{nombre_tecnico}"),
        )

    # --------------------------------------------------------
    # COMPARATIVA
    # --------------------------------------------------------

    if len(importancias) > 1:

        crear_grafico_comparativo(
            importancias,
            ruta_grafico("importancia_comparativa"),
        )

        print()
        print("=" * 60)
        print("COINCIDENCIA ENTRE MODELOS")
        print("=" * 60)

        resumen = None

        for nombre, importancia in importancias.items():

            columna = (
                importancia
                .set_index("variable_legible")["importancia"]
                .rename(nombre)
            )

            resumen = (
                columna.to_frame()
                if resumen is None
                else resumen.join(columna, how="outer")
            )

        resumen = (
            resumen
            .fillna(0)
            .sort_values(
                resumen.columns[0],
                ascending=False,
            )
            .round(2)
        )

        print()
        print(resumen.head(10).to_string())

        resumen.to_csv(
            PROCESSED_RESULTADOS_DIR
            / "importancia_comparativa.csv",
            encoding="utf-8",
        )

    print()
    print("=" * 60)
    print("ANÁLISIS DE IMPORTANCIA COMPLETADO")
    print("=" * 60)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
