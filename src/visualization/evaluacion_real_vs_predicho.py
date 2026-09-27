# src/visualization/evaluacion_real_vs_predicho.py

# ============================================================
# GRÁFICOS DE EVALUACIÓN
# ============================================================
#
# Genera las figuras que se usan en la memoria y en la
# aplicación:
#
#   1. Serie temporal de clientes reales y predichos.
#   2. Dispersión real frente a predicho.
#   3. Comparativa de los cuatro modelos.
#   4. Demanda media por día de la semana.
#   5. Relación entre meteorología y demanda.
#
# Todas parten de las predicciones ya guardadas, no de un
# modelo reentrenado aquí.
#
# ============================================================

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
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
    COMPARACION_MODELOS,
    GOLD_DATASET,
    MODELO_PRODUCCION,
    PROCESSED_RESULTADOS_DIR,
    ruta_grafico,
)


# ============================================================
# ESTILO
# ============================================================

COLOR_REAL = "#1f4e79"
COLOR_PREDICHO = "#e08a2e"
COLOR_INTERVALO = "#e08a2e"

ORDEN_DIAS = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]


def limpiar_ejes(eje):

    eje.grid(alpha=0.3)
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)


# ============================================================
# CARGA
# ============================================================

def cargar_predicciones(nombre_modelo=MODELO_PRODUCCION):

    ruta = (
        PROCESSED_RESULTADOS_DIR
        / f"predicciones_{nombre_modelo}.csv"
    )

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existen las predicciones de "
            f"'{nombre_modelo}':\n{ruta}"
        )

    return pd.read_csv(ruta, parse_dates=["fecha"])


def cargar_gold():

    if not GOLD_DATASET.exists():

        raise FileNotFoundError(
            f"No existe la capa gold: {GOLD_DATASET}"
        )

    return pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])


# ============================================================
# 1. SERIE TEMPORAL
# ============================================================

def grafico_serie_temporal(test, ruta):
    """
    Compara día a día lo que pasó con lo que el modelo había
    previsto.

    Es el gráfico más honesto de todos: se ve de un vistazo si
    el modelo sigue el ritmo real del negocio o si va por su
    cuenta.
    """

    test = test.sort_values("fecha")

    figura, eje = plt.subplots(figsize=(14, 6))

    eje.plot(
        test["fecha"],
        test["n_clientes"],
        color=COLOR_REAL,
        linewidth=1.6,
        label="Clientes reales",
    )

    eje.plot(
        test["fecha"],
        test["prediccion"],
        color=COLOR_PREDICHO,
        linewidth=1.6,
        linestyle="--",
        label="Predicción",
    )

    eje.fill_between(
        test["fecha"],
        test["n_clientes"],
        test["prediccion"],
        color="#999999",
        alpha=0.18,
        label="Diferencia",
    )

    eje.set_title(
        "Clientes reales frente a predichos · 2026",
        fontsize=14,
        pad=14,
    )

    eje.set_xlabel("Fecha")
    eje.set_ylabel("Número de clientes")

    eje.xaxis.set_major_locator(mdates.MonthLocator())
    eje.xaxis.set_major_formatter(
        mdates.DateFormatter("%b")
    )

    eje.legend(loc="upper left")

    limpiar_ejes(eje)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


# ============================================================
# 2. DISPERSIÓN
# ============================================================

def grafico_dispersion(test, ruta):
    """
    Cada punto es un día. Cuanto más cerca de la diagonal,
    mejor la predicción.
    """

    figura, eje = plt.subplots(figsize=(7.5, 7))

    eje.scatter(
        test["prediccion"],
        test["n_clientes"],
        alpha=0.6,
        s=34,
        color=COLOR_REAL,
        edgecolor="none",
    )

    maximo = float(
        max(
            test["n_clientes"].max(),
            test["prediccion"].max(),
        )
    ) * 1.05

    eje.plot(
        [0, maximo],
        [0, maximo],
        color="#c0392b",
        linestyle="--",
        linewidth=1.4,
        label="Predicción perfecta",
    )

    correlacion = float(
        np.corrcoef(test["prediccion"], test["n_clientes"])[0, 1]
    )

    eje.set_title(
        "Real frente a predicho · 2026\n"
        f"correlación = {correlacion:.3f}",
        fontsize=13,
        pad=14,
    )

    eje.set_xlabel("Clientes predichos")
    eje.set_ylabel("Clientes reales")

    eje.set_xlim(0, maximo)
    eje.set_ylim(0, maximo)

    eje.legend(loc="upper left")

    limpiar_ejes(eje)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


# ============================================================
# 3. COMPARATIVA DE MODELOS
# ============================================================

def grafico_comparacion_modelos(ruta):
    """
    Compara el error de los cuatro modelos en validación y en
    test, junto al baseline.
    """

    if not COMPARACION_MODELOS.exists():
        return None

    comparacion = pd.read_csv(COMPARACION_MODELOS)

    comparacion = comparacion.sort_values(
        "mae_validacion",
        ascending=False,
    )

    figura, eje = plt.subplots(figsize=(11, 5.5))

    posiciones = np.arange(len(comparacion))

    ancho = 0.38

    eje.barh(
        posiciones + ancho / 2,
        comparacion["mae_validacion"],
        height=ancho,
        color="#2f6f9f",
        label="Validación (2025)",
        edgecolor="white",
    )

    eje.barh(
        posiciones - ancho / 2,
        comparacion["mae_test"],
        height=ancho,
        color="#e08a2e",
        label="Test (2026)",
        edgecolor="white",
    )

    for posicion, (validacion, test) in enumerate(
        zip(
            comparacion["mae_validacion"],
            comparacion["mae_test"],
        )
    ):

        eje.text(
            validacion + 0.3,
            posicion + ancho / 2,
            f"{validacion:.1f}",
            va="center",
            fontsize=9,
        )

        eje.text(
            test + 0.3,
            posicion - ancho / 2,
            f"{test:.1f}",
            va="center",
            fontsize=9,
        )

    eje.set_yticks(posiciones)
    eje.set_yticklabels(comparacion["modelo"])

    eje.set_xlabel("Error medio absoluto (clientes)")

    eje.set_title(
        "Comparación de modelos\n"
        "menos es mejor",
        fontsize=13,
        pad=14,
    )

    eje.legend(loc="lower right")

    eje.grid(axis="x", alpha=0.3)
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)

    return comparacion


# ============================================================
# 4. DEMANDA POR DÍA DE LA SEMANA
# ============================================================

def grafico_demanda_semanal(gold, ruta):
    """
    Perfil semanal de la demanda: el patrón más fuerte del
    negocio.
    """

    resumen = (
        gold
        .groupby("nombre_dia")["n_clientes"]
        .agg(["mean", "std", "count"])
        .reindex(
            [
                dia
                for dia in ORDEN_DIAS
                if dia in gold["nombre_dia"].unique()
            ]
        )
    )

    figura, eje = plt.subplots(figsize=(10, 5.5))

    eje.bar(
        resumen.index,
        resumen["mean"],
        yerr=resumen["std"],
        capsize=5,
        color=COLOR_REAL,
        edgecolor="white",
        error_kw={"ecolor": "#7f8c8d", "linewidth": 1.2},
    )

    for posicion, valor in enumerate(resumen["mean"]):

        eje.text(
            posicion,
            valor + 4,
            f"{valor:.0f}",
            ha="center",
            fontsize=10,
        )

    eje.set_title(
        "Clientes por día de la semana\n"
        "media y desviación típica, 2022-2026",
        fontsize=13,
        pad=14,
    )

    eje.set_ylabel("Número de clientes")

    eje.grid(axis="y", alpha=0.3)
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)

    return resumen


# ============================================================
# 5. METEOROLOGÍA Y DEMANDA
# ============================================================

def grafico_meteorologia(gold, ruta):
    """
    Relación entre temperatura, lluvia y demanda.

    Se separan los fines de semana del resto porque el efecto
    de la meteorología no es el mismo: quien sale a comer un
    sábado puede quedarse en casa si llueve; quien come cerca
    del trabajo un miércoles, no.
    """

    figura, ejes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # --- Temperatura ---

    for etiqueta, mascara, color in [
        ("Entre semana", gold["fin_de_semana"] == 0, "#2f6f9f"),
        ("Fin de semana", gold["fin_de_semana"] == 1, "#e08a2e"),
    ]:

        datos = gold[mascara]

        ejes[0].scatter(
            datos["tmed"],
            datos["n_clientes"],
            alpha=0.4,
            s=18,
            color=color,
            edgecolor="none",
            label=etiqueta,
        )

    ejes[0].set_title("Temperatura media y demanda")
    ejes[0].set_xlabel("Temperatura media (°C)")
    ejes[0].set_ylabel("Número de clientes")
    ejes[0].legend()

    # --- Lluvia ---

    gold = gold.copy()

    gold["tramo_lluvia"] = pd.cut(
        gold["prec"],
        bins=[-0.01, 0.09, 2, 10, 1000],
        labels=[
            "sin lluvia",
            "< 2 mm",
            "2 - 10 mm",
            "> 10 mm",
        ],
    )

    resumen = (
        gold
        .groupby(
            ["tramo_lluvia", "fin_de_semana"],
            observed=True,
        )["n_clientes"]
        .mean()
        .unstack()
    )

    resumen.plot(
        kind="bar",
        ax=ejes[1],
        color=["#2f6f9f", "#e08a2e"],
        edgecolor="white",
    )

    ejes[1].set_title("Lluvia y demanda")
    ejes[1].set_xlabel("Precipitación acumulada")
    ejes[1].set_ylabel("Clientes (media)")
    ejes[1].tick_params(axis="x", rotation=0)

    ejes[1].legend(
        ["Entre semana", "Fin de semana"],
        title="",
    )

    for eje in ejes:
        limpiar_ejes(eje)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)

    return resumen


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRÁFICOS DE EVALUACIÓN")
    print("=" * 60)

    predicciones = cargar_predicciones()

    gold = cargar_gold()

    test = predicciones[
        predicciones["conjunto"] == "test"
    ].copy()

    print()
    print(f"Modelo:  {predicciones['modelo'].iloc[0]}")
    print(f"Días:    {len(test)} (test 2026)")

    # --------------------------------------------------------
    # GENERAR
    # --------------------------------------------------------

    print()
    print("Generando gráficos...")

    grafico_serie_temporal(
        test,
        ruta_grafico("serie_real_vs_predicho"),
    )

    print("  - serie_real_vs_predicho.png")

    grafico_dispersion(
        test,
        ruta_grafico("dispersion_real_vs_predicho"),
    )

    print("  - dispersion_real_vs_predicho.png")

    comparacion = grafico_comparacion_modelos(
        ruta_grafico("comparacion_modelos")
    )

    if comparacion is not None:
        print("  - comparacion_modelos.png")

    resumen_semanal = grafico_demanda_semanal(
        gold,
        ruta_grafico("demanda_semanal"),
    )

    print("  - demanda_semanal.png")

    resumen_meteo = grafico_meteorologia(
        gold,
        ruta_grafico("meteorologia_demanda"),
    )

    print("  - meteorologia_demanda.png")

    # --------------------------------------------------------
    # LECTURA DE LOS RESULTADOS
    # --------------------------------------------------------

    print()
    print("DEMANDA MEDIA POR DÍA DE LA SEMANA")
    print("-" * 60)

    print(
        resumen_semanal[["mean", "std", "count"]]
        .round(1)
        .rename(
            columns={
                "mean": "media",
                "std": "desviación",
                "count": "días",
            }
        )
        .to_string()
    )

    print()
    print("DEMANDA MEDIA SEGÚN LA LLUVIA")
    print("-" * 60)

    resumen_meteo.columns = [
        "entre semana",
        "fin de semana",
    ]

    print(resumen_meteo.round(1).to_string())

    seco = resumen_meteo.iloc[0]
    lluvioso = resumen_meteo.iloc[-1]

    print()

    for columna in resumen_meteo.columns:

        diferencia = (
            100 * (lluvioso[columna] - seco[columna])
            / seco[columna]
        )

        print(
            f"Un día de lluvia fuerte {columna} tiene un "
            f"{abs(diferencia):.0f} % "
            f"{'menos' if diferencia < 0 else 'más'} "
            "de clientes que uno seco."
        )

    print()
    print("=" * 60)
    print("GRÁFICOS GENERADOS CORRECTAMENTE")
    print("=" * 60)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
