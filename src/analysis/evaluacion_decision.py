# src/analysis/evaluacion_decision.py

# ============================================================
# EVALUACIÓN COMO SISTEMA DE DECISIÓN
# ============================================================
#
# Todo el resto del proyecto mide el error en clientes. Pero el
# encargado no decide clientes: decide personas. La pregunta
# que de verdad importa no es "¿cuánto se equivoca el modelo?"
# sino "¿habría acertado con la plantilla?".
#
# Este módulo la responde. Para cada día de 2026 compara tres
# formas de decidir el personal:
#
#   1. INTUICIÓN   -> el baseline: la media histórica de ese
#                     día de la semana. Es lo que hace hoy un
#                     encargado con experiencia y sin
#                     herramientas.
#
#   2. MODELO      -> la recomendación del sistema.
#
#   3. ORÁCULO     -> la plantilla que se habría puesto
#                     sabiendo de antemano los clientes que
#                     iban a venir. No es alcanzable: marca el
#                     suelo teórico contra el que medirse.
#
# ------------------------------------------------------------
# LOS DOS ERRORES NO CUESTAN LO MISMO
# ------------------------------------------------------------
#
# Quedarse corto de personal y pasarse son fallos distintos y
# hay que contarlos por separado:
#
#   - PASARSE cuesta salario. Es un coste conocido, acotado y
#     que se paga una vez.
#
#   - QUEDARSE CORTO cuesta ventas y reputación. Es un coste
#     difuso, mayor, y con efectos que duran más de un día.
#
# Un sistema que reduce el error medio pero se queda corto más
# a menudo puede ser peor para el negocio que uno menos
# preciso. Por eso el MAE, por sí solo, no basta para decidir.
#
# ------------------------------------------------------------
# CÓMO SE DEFINE "ACERTAR"
# ------------------------------------------------------------
#
# Con los mismos umbrales con los que se etiqueta la jornada en
# los datos de actividad:
#
#   plantilla mínima  = ceil(clientes / 24)   por debajo falta
#                                             personal
#   plantilla máxima  = ceil(clientes / 15)   por encima sobra
#
# Entre ambas, la jornada sale bien. No se exige clavar el
# número: se exige caer dentro de la banda cómoda.
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
    COSTE_JORNADA_EMPLEADO,
    EMPLEADOS_MAXIMO,
    EMPLEADOS_MINIMO,
    FRACCION_DEMANDA_PERDIDA,
    MARGEN_POR_CLIENTE,
    MODELO_PRODUCCION,
    PROCESSED_RESULTADOS_DIR,
    RATIO_FALTA_PERSONAL,
    RATIO_PERSONAL_OCUPADO,
    ruta_grafico,
)

from src.models.prediccion import recomendar_empleados


SALIDA = PROCESSED_RESULTADOS_DIR / "evaluacion_decision.csv"

SALIDA_SENSIBILIDAD = (
    PROCESSED_RESULTADOS_DIR / "sensibilidad_costes.csv"
)


# ============================================================
# CARGA
# ============================================================

def cargar_predicciones(nombre_modelo):
    """
    Carga las predicciones de test de un modelo.
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
            "    python pipeline.py"
        )

    datos = pd.read_csv(ruta, parse_dates=["fecha"])

    return datos[datos["conjunto"] == "test"].copy()


# ============================================================
# BANDAS DE PLANTILLA CORRECTA
# ============================================================

def plantilla_minima(clientes):
    """
    Por debajo de esta plantilla falta personal.
    """

    minima = np.ceil(
        np.asarray(clientes, dtype=float)
        / RATIO_FALTA_PERSONAL
    )

    return np.clip(minima, EMPLEADOS_MINIMO, EMPLEADOS_MAXIMO)


def plantilla_maxima(clientes):
    """
    Por encima de esta plantilla sobra personal.
    """

    maxima = np.ceil(
        np.asarray(clientes, dtype=float)
        / RATIO_PERSONAL_OCUPADO
    )

    return np.clip(maxima, EMPLEADOS_MINIMO, EMPLEADOS_MAXIMO)


# ============================================================
# COSTE DE UNA DECISIÓN
# ============================================================

def coste_infradotacion(
    clientes_reales,
    empleados,
    fraccion_perdida=FRACCION_DEMANDA_PERDIDA,
    margen=MARGEN_POR_CLIENTE,
):
    """
    Coste de quedarse corto de personal.

    Con `empleados` en sala se pueden atender como mucho
    `empleados x 24` clientes. Lo que exceda esa capacidad se
    pierde en parte: gente que se va sin sentarse, que no
    repite o que consume menos porque el servicio se resiente.

    `fraccion_perdida` es el parámetro más incierto del
    modelo económico y por eso el análisis lo recorre en un
    rango amplio.
    """

    clientes_reales = np.asarray(clientes_reales, dtype=float)
    empleados = np.asarray(empleados, dtype=float)

    capacidad = empleados * RATIO_FALTA_PERSONAL

    exceso = np.maximum(clientes_reales - capacidad, 0)

    return exceso * fraccion_perdida * margen


def coste_sobredotacion(
    clientes_reales,
    empleados,
    coste_jornada=COSTE_JORNADA_EMPLEADO,
):
    """
    Coste de pasarse de personal: las jornadas pagadas por
    encima de lo que la demanda necesitaba.
    """

    empleados = np.asarray(empleados, dtype=float)

    sobrantes = np.maximum(
        empleados - plantilla_maxima(clientes_reales),
        0,
    )

    return sobrantes * coste_jornada


# ============================================================
# EVALUACIÓN DE UN MÉTODO
# ============================================================

def evaluar_metodo(
    clientes_reales,
    clientes_estimados,
    fraccion_perdida=FRACCION_DEMANDA_PERDIDA,
):
    """
    Convierte una estimación de clientes en una decisión de
    plantilla y mide qué habría pasado.
    """

    empleados = np.array([
        recomendar_empleados(estimacion)
        for estimacion in clientes_estimados
    ], dtype=float)

    minima = plantilla_minima(clientes_reales)
    maxima = plantilla_maxima(clientes_reales)

    infradotado = empleados < minima
    sobredotado = empleados > maxima

    correcto = ~infradotado & ~sobredotado

    coste_falta = np.where(
        infradotado,
        coste_infradotacion(
            clientes_reales,
            empleados,
            fraccion_perdida,
        ),
        0.0,
    )

    coste_sobra = np.where(
        sobredotado,
        coste_sobredotacion(clientes_reales, empleados),
        0.0,
    )

    return {
        "empleados": empleados,
        "infradotado": infradotado,
        "sobredotado": sobredotado,
        "correcto": correcto,
        "coste_falta": coste_falta,
        "coste_sobra": coste_sobra,
        "coste_total": coste_falta + coste_sobra,
    }


def resumir(nombre, resultado, dias):
    """
    Resume la evaluación de un método.
    """

    coste_total = float(resultado["coste_total"].sum())

    return {
        "metodo": nombre,
        "dias": dias,

        "dias_correctos": int(resultado["correcto"].sum()),
        "dias_infradotados": int(resultado["infradotado"].sum()),
        "dias_sobredotados": int(resultado["sobredotado"].sum()),

        "pct_correctos": round(
            100 * resultado["correcto"].mean(), 1
        ),
        "pct_infradotados": round(
            100 * resultado["infradotado"].mean(), 1
        ),
        "pct_sobredotados": round(
            100 * resultado["sobredotado"].mean(), 1
        ),

        "coste_falta_personal": round(
            float(resultado["coste_falta"].sum()), 0
        ),
        "coste_exceso_personal": round(
            float(resultado["coste_sobra"].sum()), 0
        ),
        "coste_total": round(coste_total, 0),
        "coste_por_dia": round(coste_total / dias, 1),

        # Extrapolación a un año de actividad. El restaurante
        # abre unos 245 días al año.
        "coste_anual_estimado": round(
            coste_total / dias * 245, 0
        ),
    }


# ============================================================
# SENSIBILIDAD
# ============================================================

def analizar_sensibilidad(
    clientes_reales,
    estimaciones,
    fracciones=(0.25, 0.50, 0.75, 1.00),
    costes_jornada=(90.0, 119.0, 150.0),
):
    """
    Repite el análisis variando los dos supuestos económicos
    más discutibles.

    Si la conclusión se mantiene en todo el rango, deja de
    depender de unos números que no son del establecimiento.
    """

    filas = []

    for fraccion in fracciones:

        for coste_jornada in costes_jornada:

            costes = {}

            for nombre, estimacion in estimaciones.items():

                empleados = np.array([
                    recomendar_empleados(valor)
                    for valor in estimacion
                ], dtype=float)

                minima = plantilla_minima(clientes_reales)
                maxima = plantilla_maxima(clientes_reales)

                falta = np.where(
                    empleados < minima,
                    coste_infradotacion(
                        clientes_reales,
                        empleados,
                        fraccion,
                    ),
                    0.0,
                )

                sobra = np.where(
                    empleados > maxima,
                    np.maximum(empleados - maxima, 0)
                    * coste_jornada,
                    0.0,
                )

                costes[nombre] = float((falta + sobra).sum())

            ahorro = costes["Intuición"] - costes["Modelo"]

            filas.append({
                "fraccion_perdida": fraccion,
                "coste_jornada": coste_jornada,
                "coste_intuicion": round(costes["Intuición"], 0),
                "coste_modelo": round(costes["Modelo"], 0),
                "ahorro": round(ahorro, 0),
                "ahorro_pct": round(
                    100 * ahorro / costes["Intuición"], 1
                ) if costes["Intuición"] else 0.0,
                "modelo_mejor": ahorro > 0,
            })

    return pd.DataFrame(filas)


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_decisiones(resumenes, ruta):
    """
    Reparto de días correctos, cortos y sobrados por método.
    """

    tabla = pd.DataFrame(resumenes).set_index("metodo")

    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Reparto de días ---

    posiciones = np.arange(len(tabla))

    ejes[0].bar(
        posiciones,
        tabla["pct_correctos"],
        label="Plantilla adecuada",
        color="#27ae60",
        edgecolor="white",
    )

    ejes[0].bar(
        posiciones,
        tabla["pct_infradotados"],
        bottom=tabla["pct_correctos"],
        label="Falta de personal",
        color="#c0392b",
        edgecolor="white",
    )

    ejes[0].bar(
        posiciones,
        tabla["pct_sobredotados"],
        bottom=tabla["pct_correctos"] + tabla["pct_infradotados"],
        label="Exceso de personal",
        color="#d68910",
        edgecolor="white",
    )

    ejes[0].set_xticks(posiciones)
    ejes[0].set_xticklabels(tabla.index)
    ejes[0].set_ylabel("% de días")
    ejes[0].set_title("Cómo habría salido la jornada")
    ejes[0].legend(loc="lower right", fontsize=9)

    # --- Coste ---

    ejes[1].bar(
        posiciones,
        tabla["coste_falta_personal"],
        label="Ventas perdidas",
        color="#c0392b",
        edgecolor="white",
    )

    ejes[1].bar(
        posiciones,
        tabla["coste_exceso_personal"],
        bottom=tabla["coste_falta_personal"],
        label="Salario de más",
        color="#d68910",
        edgecolor="white",
    )

    for posicion, total in enumerate(tabla["coste_total"]):

        ejes[1].text(
            posicion,
            total + tabla["coste_total"].max() * 0.03,
            f"{total:,.0f} €".replace(",", "."),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ejes[1].set_xticks(posiciones)
    ejes[1].set_xticklabels(tabla.index)
    ejes[1].set_ylabel("Coste acumulado (€)")
    ejes[1].set_title(
        "Coste de las decisiones equivocadas\n"
        "enero a julio de 2026"
    )
    ejes[1].legend(loc="upper right", fontsize=9)

    for eje in ejes:
        eje.grid(axis="y", alpha=0.3)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


def grafico_sensibilidad(sensibilidad, ruta):
    """
    Ahorro del modelo sobre la intuición en todo el rango de
    supuestos.
    """

    figura, eje = plt.subplots(figsize=(11, 5.5))

    for coste_jornada, grupo in sensibilidad.groupby(
        "coste_jornada"
    ):

        eje.plot(
            grupo["fraccion_perdida"],
            grupo["ahorro"],
            marker="o",
            linewidth=2,
            label=f"{coste_jornada:.0f} € por jornada",
        )

    eje.axhline(
        0,
        color="#c0392b",
        linestyle="--",
        linewidth=1.4,
    )

    eje.set_xlabel(
        "Fracción de la demanda excedente que se pierde "
        "de verdad"
    )

    eje.set_ylabel("Ahorro frente a la intuición (€)")

    eje.set_title(
        "El ahorro se mantiene en todo el rango de supuestos\n"
        "enero a julio de 2026",
        fontsize=13,
        pad=14,
    )

    eje.legend(title="Coste de una jornada")

    eje.grid(alpha=0.3)
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
    print("EVALUACIÓN COMO SISTEMA DE DECISIÓN")
    print("=" * 60)

    # --------------------------------------------------------
    # DATOS
    # --------------------------------------------------------

    modelo = cargar_predicciones(MODELO_PRODUCCION)

    baseline = cargar_predicciones("baseline")

    # Se alinean por fecha para comparar el mismo día.
    datos = modelo.merge(
        baseline[["fecha", "prediccion"]].rename(
            columns={"prediccion": "prediccion_baseline"}
        ),
        on="fecha",
        how="inner",
        validate="one_to_one",
    )

    clientes_reales = datos["n_clientes"].to_numpy(dtype=float)

    dias = len(datos)

    print()
    print(f"Días evaluados: {dias}")
    print(
        f"Periodo:        {datos['fecha'].min().date()} -> "
        f"{datos['fecha'].max().date()}"
    )

    # --------------------------------------------------------
    # SUPUESTOS
    # --------------------------------------------------------

    print()
    print("SUPUESTOS ECONÓMICOS")
    print("-" * 60)
    print(
        f"  Coste de una jornada de camarero:  "
        f"{COSTE_JORNADA_EMPLEADO:.0f} €"
    )
    print(
        f"  Margen de contribución por cliente: "
        f"{MARGEN_POR_CLIENTE:.1f} €"
    )
    print(
        f"  Demanda excedente que se pierde:    "
        f"{100 * FRACCION_DEMANDA_PERDIDA:.0f} %"
    )
    print()
    print(
        "  Son referencias del sector, no datos del "
        "establecimiento."
    )
    print(
        "  Por eso al final se recorre todo el rango."
    )

    # --------------------------------------------------------
    # LOS TRES MÉTODOS
    # --------------------------------------------------------

    estimaciones = {
        "Intuición": datos["prediccion_baseline"].to_numpy(
            dtype=float
        ),
        "Modelo": datos["prediccion"].to_numpy(dtype=float),
        "Oráculo": clientes_reales,
    }

    resultados = {
        nombre: evaluar_metodo(clientes_reales, estimacion)
        for nombre, estimacion in estimaciones.items()
    }

    resumenes = [
        resumir(nombre, resultado, dias)
        for nombre, resultado in resultados.items()
    ]

    tabla = pd.DataFrame(resumenes)

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("¿HABRÍA ACERTADO CON LA PLANTILLA?")
    print("=" * 60)
    print()

    print(
        f"{'':<12} {'correctos':>11} {'se queda corto':>16} "
        f"{'se pasa':>10}"
    )

    print("-" * 60)

    for fila in resumenes:

        print(
            f"{fila['metodo']:<12} "
            f"{fila['dias_correctos']:>4} "
            f"({fila['pct_correctos']:>4.1f} %) "
            f"{fila['dias_infradotados']:>6} "
            f"({fila['pct_infradotados']:>4.1f} %) "
            f"{fila['dias_sobredotados']:>3} "
            f"({fila['pct_sobredotados']:>4.1f} %)"
        )

    # --------------------------------------------------------
    # COSTE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CUÁNTO CUESTAN LAS DECISIONES EQUIVOCADAS")
    print("=" * 60)
    print()

    print(
        f"{'':<12} {'ventas perdidas':>17} "
        f"{'salario de más':>16} {'total':>12} {'al año':>12}"
    )

    print("-" * 74)

    for fila in resumenes:

        print(
            f"{fila['metodo']:<12} "
            f"{fila['coste_falta_personal']:>15,.0f} € "
            f"{fila['coste_exceso_personal']:>14,.0f} € "
            f"{fila['coste_total']:>10,.0f} € "
            f"{fila['coste_anual_estimado']:>10,.0f} €"
        )

    # --------------------------------------------------------
    # LO QUE APORTA EL MODELO
    # --------------------------------------------------------

    intuicion = tabla[tabla["metodo"] == "Intuición"].iloc[0]
    con_modelo = tabla[tabla["metodo"] == "Modelo"].iloc[0]
    oraculo = tabla[tabla["metodo"] == "Oráculo"].iloc[0]

    ahorro = (
        intuicion["coste_total"] - con_modelo["coste_total"]
    )

    ahorro_anual = (
        intuicion["coste_anual_estimado"]
        - con_modelo["coste_anual_estimado"]
    )

    margen_teorico = (
        intuicion["coste_total"] - oraculo["coste_total"]
    )

    print()
    print("=" * 60)
    print("LO QUE APORTA EL MODELO")
    print("=" * 60)

    print()
    print(
        f"  Ahorro en el periodo evaluado: "
        f"{ahorro:,.0f} €"
    )

    print(
        f"  Ahorro anual estimado:         "
        f"{ahorro_anual:,.0f} €"
    )

    if margen_teorico > 0:

        print(
            f"  Captura el "
            f"{100 * ahorro / margen_teorico:.0f} % de la "
            "mejora que sería posible con predicción perfecta."
        )

    dias_salvados = (
        intuicion["dias_infradotados"]
        - con_modelo["dias_infradotados"]
    )

    print()

    if dias_salvados > 0:

        print(
            f"  Evita {int(dias_salvados)} días de falta de "
            "personal en siete meses"
        )

        print(
            f"  ({int(dias_salvados * 245 / dias)} al año)."
        )

    print()
    print(
        "  El oráculo marca el suelo: es el coste que queda "
        "aunque se acierte"
    )

    print(
        "  la demanda exactamente, porque la plantilla solo "
        "se mueve en"
    )

    print("  números enteros de personas.")

    # --------------------------------------------------------
    # SENSIBILIDAD
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SENSIBILIDAD A LOS SUPUESTOS")
    print("=" * 60)

    sensibilidad = analizar_sensibilidad(
        clientes_reales,
        {
            nombre: estimacion
            for nombre, estimacion in estimaciones.items()
            if nombre != "Oráculo"
        },
    )

    print()
    print(
        sensibilidad[
            [
                "fraccion_perdida",
                "coste_jornada",
                "coste_intuicion",
                "coste_modelo",
                "ahorro",
                "ahorro_pct",
            ]
        ].to_string(index=False)
    )

    escenarios_favorables = int(
        sensibilidad["modelo_mejor"].sum()
    )

    print()

    if escenarios_favorables == len(sensibilidad):

        print(
            f"El modelo sale mejor en los "
            f"{len(sensibilidad)} escenarios probados."
        )

        print(
            "La conclusión no depende de los supuestos "
            "económicos concretos."
        )

    else:

        print(
            f"El modelo sale mejor en "
            f"{escenarios_favorables} de "
            f"{len(sensibilidad)} escenarios."
        )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    detalle = datos[
        ["fecha", "dia_semana", "n_clientes", "prediccion"]
    ].copy()

    detalle["prediccion_baseline"] = datos[
        "prediccion_baseline"
    ]

    for nombre, resultado in resultados.items():

        clave = nombre.lower().replace("ó", "o").replace(
            "á", "a"
        )

        detalle[f"empleados_{clave}"] = resultado["empleados"]
        detalle[f"coste_{clave}"] = resultado["coste_total"]

    detalle.to_csv(SALIDA, index=False, encoding="utf-8")

    tabla.to_csv(
        PROCESSED_RESULTADOS_DIR / "resumen_decision.csv",
        index=False,
        encoding="utf-8",
    )

    sensibilidad.to_csv(
        SALIDA_SENSIBILIDAD,
        index=False,
        encoding="utf-8",
    )

    grafico_decisiones(
        resumenes,
        ruta_grafico("evaluacion_decision"),
    )

    grafico_sensibilidad(
        sensibilidad,
        ruta_grafico("sensibilidad_costes"),
    )

    print()
    print(f"Detalle diario:  {SALIDA}")
    print(f"Sensibilidad:    {SALIDA_SENSIBILIDAD}")

    print()
    print("=" * 60)
    print("EVALUACIÓN DE LA DECISIÓN COMPLETADA")
    print("=" * 60)

    return tabla, sensibilidad


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
