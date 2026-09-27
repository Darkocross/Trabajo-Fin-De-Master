# src/visualization/analisis_errores.py

# ============================================================
# ANÁLISIS DE ERRORES
# ============================================================
#
# Una métrica global esconde tanto como enseña. Un MAE de 15
# clientes puede significar que el modelo falla 15 clientes
# todos los días, o que acierta casi siempre y se equivoca 80
# en cuatro sábados de agosto. Para un restaurante son dos
# situaciones muy distintas.
#
# Este módulo abre el error y responde a:
#
#   - ¿Se equivoca más en unos días de la semana que en otros?
#   - ¿En qué meses falla más?
#   - ¿Se queda corto o se pasa de forma sistemática?
#   - ¿Cuáles fueron los peores días y qué tenían en común?
#   - ¿Empeora cuando llueve o cuando hace mucho calor?
#
# NO reentrena nada: lee las predicciones que ya generó el
# modelo. Antes este script entrenaba su propia copia del
# modelo con otras variables, de modo que los errores que
# analizaba no eran los del modelo evaluado.
#
# ============================================================

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    ANALISIS_ERRORES,
    MODELO_PRODUCCION,
    PROCESSED_RESULTADOS_DIR,
    ruta_grafico,
)

from src.models.features import NOMBRES_DIAS, NOMBRES_MESES


# ============================================================
# CARGA
# ============================================================

def cargar_predicciones(nombre_modelo=MODELO_PRODUCCION):
    """
    Carga las predicciones guardadas por un modelo.
    """

    ruta = (
        PROCESSED_RESULTADOS_DIR
        / f"predicciones_{nombre_modelo}.csv"
    )

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existen las predicciones de "
            f"'{nombre_modelo}':\n{ruta}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py --hasta boosting"
        )

    predicciones = pd.read_csv(
        ruta,
        parse_dates=["fecha"],
    )

    predicciones["nombre_dia"] = (
        predicciones["dia_semana"]
        .astype(str)
        .map(NOMBRES_DIAS)
    )

    predicciones["nombre_mes"] = (
        predicciones["mes"]
        .astype(str)
        .map(NOMBRES_MESES)
    )

    return predicciones


# ============================================================
# ANÁLISIS POR GRUPOS
# ============================================================

def error_por(predicciones, columna, orden=None):
    """
    Calcula el error medio agrupado por una variable.
    """

    resumen = (
        predicciones
        .groupby(columna)
        .agg(
            dias=("error", "size"),
            clientes_reales=("n_clientes", "mean"),
            prediccion=("prediccion", "mean"),
            error_medio=("error", "mean"),
            error_absoluto=("error_absoluto", "mean"),
        )
        .round(1)
    )

    if orden:

        resumen = resumen.reindex(
            [valor for valor in orden if valor in resumen.index]
        )

    return resumen


def analizar_sesgo(predicciones):
    """
    Comprueba si el modelo se queda corto o se pasa de forma
    sistemática.

    Distinguirlo importa: un error aleatorio se compensa a lo
    largo del mes, pero un sesgo constante significa que el
    restaurante va a estar mal dimensionado SIEMPRE en la misma
    dirección.
    """

    error_medio = predicciones["error"].mean()

    error_absoluto = predicciones["error_absoluto"].mean()

    # Proporción del error total que es sesgo y no ruido.
    proporcion = (
        abs(error_medio) / error_absoluto
        if error_absoluto
        else 0
    )

    subestimados = int((predicciones["error"] > 0).sum())

    total = len(predicciones)

    return {
        "error_medio": float(error_medio),
        "error_absoluto": float(error_absoluto),
        "proporcion_sesgo": float(proporcion),
        "dias_subestimados": subestimados,
        "porcentaje_subestimados": (
            100 * subestimados / total if total else 0
        ),
    }


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_distribucion_errores(predicciones, ruta):
    """
    Histograma del error y error frente a la demanda real.
    """

    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Histograma ---

    ejes[0].hist(
        predicciones["error"],
        bins=30,
        color="#2f6f9f",
        edgecolor="white",
    )

    ejes[0].axvline(
        0,
        color="#333333",
        linestyle="--",
        linewidth=1,
    )

    ejes[0].axvline(
        predicciones["error"].mean(),
        color="#c0392b",
        linewidth=2,
        label=(
            f"Error medio: "
            f"{predicciones['error'].mean():+.1f}"
        ),
    )

    ejes[0].set_title("Distribución del error")

    ejes[0].set_xlabel("Clientes reales - predichos")

    ejes[0].set_ylabel("Número de días")

    ejes[0].legend()

    # --- Error frente a demanda ---

    ejes[1].scatter(
        predicciones["n_clientes"],
        predicciones["error"],
        alpha=0.55,
        s=26,
        color="#2f6f9f",
        edgecolor="none",
    )

    ejes[1].axhline(
        0,
        color="#333333",
        linestyle="--",
        linewidth=1,
    )

    ejes[1].set_title("Error según el volumen del día")

    ejes[1].set_xlabel("Clientes reales")

    ejes[1].set_ylabel("Clientes reales - predichos")

    for eje in ejes:
        eje.grid(alpha=0.3)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


def grafico_error_por_grupo(
    resumen_dia,
    resumen_mes,
    ruta,
):
    """
    Error absoluto medio por día de la semana y por mes.
    """

    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))

    ejes[0].bar(
        resumen_dia.index,
        resumen_dia["error_absoluto"],
        color="#2f6f9f",
        edgecolor="white",
    )

    ejes[0].set_title("Error medio por día de la semana")

    ejes[0].set_ylabel("Error medio (clientes)")

    ejes[0].tick_params(axis="x", rotation=45)

    ejes[1].bar(
        resumen_mes.index,
        resumen_mes["error_absoluto"],
        color="#5aa469",
        edgecolor="white",
    )

    ejes[1].set_title("Error medio por mes")

    ejes[1].set_ylabel("Error medio (clientes)")

    ejes[1].tick_params(axis="x", rotation=45)

    for eje in ejes:
        eje.grid(axis="y", alpha=0.3)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ANÁLISIS DE ERRORES")
    print("=" * 60)

    predicciones = cargar_predicciones()

    print()
    print(
        f"Modelo analizado: "
        f"{predicciones['modelo'].iloc[0]}"
    )

    # --------------------------------------------------------
    # SE ANALIZA EL TEST
    # --------------------------------------------------------
    #
    # Es el conjunto que el modelo no ha visto nunca, así que
    # es donde el error se parece al que tendría funcionando
    # de verdad.

    test = predicciones[
        predicciones["conjunto"] == "test"
    ].copy()

    print(f"Días analizados: {len(test)} (test)")

    print(
        f"Periodo:         {test['fecha'].min().date()} -> "
        f"{test['fecha'].max().date()}"
    )

    # --------------------------------------------------------
    # SESGO
    # --------------------------------------------------------

    sesgo = analizar_sesgo(test)

    print()
    print("SESGO DEL MODELO")
    print("-" * 60)

    print(
        f"Error medio:            "
        f"{sesgo['error_medio']:+.1f} clientes"
    )

    print(
        f"Error absoluto medio:   "
        f"{sesgo['error_absoluto']:.1f} clientes"
    )

    print(
        f"Días subestimados:      "
        f"{sesgo['dias_subestimados']} de {len(test)} "
        f"({sesgo['porcentaje_subestimados']:.0f} %)"
    )

    print()

    if sesgo["proporcion_sesgo"] > 0.4:

        print(
            "El error NO es simétrico. El modelo se queda "
            "corto de forma sistemática: alrededor del "
            f"{100 * sesgo['proporcion_sesgo']:.0f} % del "
            "error total es sesgo, no ruido."
        )

        print()

        print(
            "Causa: el negocio crece un 4 % al año y los "
            "modelos de árboles no pueden extrapolar una "
            "tendencia fuera del rango que han visto. "
            "Solución operativa: reentrenar cada pocos meses."
        )

    else:

        print(
            "El error es razonablemente simétrico: el modelo "
            "no tiende a quedarse corto ni a pasarse."
        )

    # --------------------------------------------------------
    # POR DÍA DE LA SEMANA
    # --------------------------------------------------------

    orden_dias = [
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    ]

    resumen_dia = error_por(test, "nombre_dia", orden_dias)

    print()
    print("ERROR POR DÍA DE LA SEMANA")
    print("-" * 60)
    print(resumen_dia.to_string())

    print()

    peor_dia = resumen_dia["error_absoluto"].idxmax()

    print(
        f"El modelo falla más los {peor_dia} "
        f"({resumen_dia.loc[peor_dia, 'error_absoluto']:.1f} "
        "clientes de error medio), que son también los días "
        "de más volumen: cuanta más gente, más margen de error "
        "en términos absolutos."
    )

    # --------------------------------------------------------
    # POR MES
    # --------------------------------------------------------

    orden_meses = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ]

    resumen_mes = error_por(test, "nombre_mes", orden_meses)

    print()
    print("ERROR POR MES")
    print("-" * 60)
    print(resumen_mes.to_string())

    # --------------------------------------------------------
    # METEOROLOGÍA
    # --------------------------------------------------------

    test["condicion"] = np.select(
        [
            test["prec"] >= 5,
            test["prec"] > 0,
            test["tmax"] >= 32,
            test["tmin"] <= 5,
        ],
        [
            "lluvia apreciable",
            "lluvia débil",
            "calor intenso",
            "frío intenso",
        ],
        default="condiciones normales",
    )

    resumen_meteo = error_por(test, "condicion")

    print()
    print("ERROR SEGÚN LA METEOROLOGÍA")
    print("-" * 60)
    print(resumen_meteo.to_string())

    # --------------------------------------------------------
    # PEORES DÍAS
    # --------------------------------------------------------

    peores = (
        test
        .nlargest(10, "error_absoluto")
        [
            [
                "fecha",
                "nombre_dia",
                "n_clientes",
                "prediccion",
                "error",
                "tmax",
                "prec",
                "es_festivo",
            ]
        ]
        .round(1)
    )

    print()
    print("LOS DIEZ DÍAS CON MAYOR ERROR")
    print("-" * 60)
    print(peores.to_string(index=False))

    # --------------------------------------------------------
    # GRÁFICOS
    # --------------------------------------------------------

    grafico_distribucion_errores(
        test,
        ruta_grafico("distribucion_errores"),
    )

    grafico_error_por_grupo(
        resumen_dia,
        resumen_mes,
        ruta_grafico("error_por_grupo"),
    )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test.to_csv(
        ANALISIS_ERRORES,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Detalle guardado en: {ANALISIS_ERRORES}")

    print()
    print("=" * 60)
    print("ANÁLISIS DE ERRORES COMPLETADO")
    print("=" * 60)

    return test


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
