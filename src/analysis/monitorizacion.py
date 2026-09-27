# src/analysis/monitorizacion.py

# ============================================================
# MONITORIZACIÓN DEL MODELO EN PRODUCCIÓN
# ============================================================
#
# Un modelo entrenado no es un sistema. Lo que separa una cosa
# de la otra es saber CUÁNDO hay que volver a entrenarlo, y
# tener un criterio escrito en lugar de una intuición.
#
# La memoria decía "se corrige reentrenando cada pocos meses".
# "Cada pocos meses" no es un criterio: no dice qué mirar, ni
# a partir de qué valor actuar, ni quién se entera.
#
# ------------------------------------------------------------
# QUÉ SE VIGILA
# ------------------------------------------------------------
#
# No el error medio. El error medio de este modelo es de unos
# 18 clientes y va a seguir siéndolo: es ruido irreducible del
# problema, y verlo subir y bajar no dice nada accionable.
#
# Lo que se vigila es el SESGO, por dos motivos:
#
#   1. Es lo que se degrada con el tiempo. El negocio crece y
#      el modelo no lo sabe, así que se queda corto cada vez
#      más. El error medio apenas se mueve; el sesgo, sí.
#
#   2. Es lo que acaba costando dinero. Un error que se reparte
#      a los dos lados se compensa; uno que siempre va en la
#      misma dirección deja al local sin personal de forma
#      sistemática.
#
# ------------------------------------------------------------
# DÓNDE ESTÁ EL UMBRAL
# ------------------------------------------------------------
#
# Y aquí está lo que hace que este módulo tenga sentido, que
# salió de src/analysis/correccion_tendencia.py: el sesgo no
# importa mientras la DECISIÓN aguante.
#
# La plantilla se decide en personas enteras, y acertar
# significa caer en una banda de entre 15 y 24 clientes por
# empleado. Un sesgo de 10 clientes es aproximadamente medio
# empleado: casi siempre cabe dentro de la banda. De hecho,
# corregirlo empeora la decisión.
#
# Así que el umbral no se pone sobre el sesgo en clientes, que
# sería arbitrario, sino sobre lo que el sesgo le hace a la
# plantilla. Se avisa cuando el sesgo medio del último periodo
# equivale a MEDIO EMPLEADO o más, que es cuando empieza a
# sacar días de la banda.
#
# Con la banda actual, medio empleado son unos 9 clientes.
#
# ------------------------------------------------------------
# CÓMO SE USA
# ------------------------------------------------------------
#
#   python pipeline.py --solo monitorizacion
#
# Sobre los datos del proyecto compara el modelo desplegado con
# lo que realmente pasó. En un despliegue real se ejecutaría
# cada semana sobre las predicciones archivadas.
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


from src.config import (
    MODELO_PRODUCCION,
    PROCESSED_RESULTADOS_DIR,
    RATIO_CLIENTES_EMPLEADO_OBJETIVO,
)

from src.models.entrenamiento import calcular_metricas


SALIDA = PROCESSED_RESULTADOS_DIR / "monitorizacion.csv"


# Ventana de seguimiento. Cuatro semanas: suficiente para que
# el ruido diario se promedie, corto para enterarse pronto.

DIAS_VENTANA = 28


# Umbral de aviso, en clientes. Medio empleado.

UMBRAL_SESGO = RATIO_CLIENTES_EMPLEADO_OBJETIVO / 2


# Días mínimos para que una ventana cuente.
#
# El local cierra lunes, martes y todo agosto, así que las
# ventanas de los bordes y las que caen sobre el cierre de
# verano quedan con muy pocos días. Una media de tres días da
# un sesgo enorme que no significa nada, y dispararía avisos
# falsos justo cuando nadie está mirando.

DIAS_MINIMOS = 10


# ============================================================
# CÁLCULO
# ============================================================

def seguimiento(predicciones, dias=DIAS_VENTANA):
    """
    Calcula el sesgo y el error en ventanas consecutivas.
    """

    datos = predicciones.copy()

    datos["fecha"] = pd.to_datetime(datos["fecha"])

    datos = datos.sort_values("fecha").reset_index(drop=True)

    datos["error"] = datos["n_clientes"] - datos["prediccion"]

    origen = datos["fecha"].min()

    datos["ventana"] = (
        (datos["fecha"] - origen).dt.days // dias
    )

    resumen = (
        datos
        .groupby("ventana")
        .agg(
            desde=("fecha", "min"),
            hasta=("fecha", "max"),
            dias=("fecha", "size"),
            sesgo=("error", "mean"),
            error_absoluto=("error", lambda x: x.abs().mean()),
        )
        .reset_index(drop=True)
    )

    resumen["sesgo"] = resumen["sesgo"].round(2)

    resumen["error_absoluto"] = (
        resumen["error_absoluto"].round(2)
    )

    resumen["empleados_de_sesgo"] = (
        resumen["sesgo"] / RATIO_CLIENTES_EMPLEADO_OBJETIVO
    ).round(2)

    resumen["fiable"] = resumen["dias"] >= DIAS_MINIMOS

    resumen["aviso"] = (
        (resumen["sesgo"].abs() >= UMBRAL_SESGO)
        & resumen["fiable"]
    )

    return resumen


def diagnostico(resumen, ventanas=3):
    """
    Decide si hay que reentrenar.

    Mirar solo la última ventana daría falsos positivos con
    cualquier racha rara, y exigir que TODAS las últimas estén
    en aviso daría falsos negativos: basta con que una ventana
    tranquila se cuele en medio para tapar una deriva evidente.

    El criterio es el sesgo ACUMULADO de las últimas ventanas
    fiables, ponderado por días, más la condición de que todas
    apunten en la misma dirección. Así una ventana floja no
    borra la señal, pero una alternancia de signos sí la
    descarta, que es lo que la distingue del ruido.
    """

    fiables = resumen[resumen["fiable"]]

    if len(fiables) < ventanas:

        return False, (
            f"Solo hay {len(fiables)} ventanas fiables: "
            f"hacen falta {ventanas} para decidir."
        )

    ultimas = fiables.tail(ventanas)

    # Sesgo acumulado, ponderado por días de cada ventana.

    sesgo = float(
        np.average(
            ultimas["sesgo"],
            weights=ultimas["dias"],
        )
    )

    mismo_signo = len(set(np.sign(ultimas["sesgo"]))) == 1

    empleados = sesgo / RATIO_CLIENTES_EMPLEADO_OBJETIVO

    if abs(sesgo) < UMBRAL_SESGO:

        return False, (
            f"Sesgo acumulado de las últimas {ventanas} "
            f"ventanas: {sesgo:+.2f} clientes "
            f"({empleados:+.2f} empleados). Por debajo del "
            f"umbral: la banda de plantilla lo absorbe."
        )

    if not mismo_signo:

        return False, (
            f"Sesgo acumulado alto ({sesgo:+.2f}) pero las "
            f"ventanas no coinciden en dirección: es ruido, "
            f"no deriva."
        )

    direccion = (
        "se queda corto" if sesgo > 0 else "se pasa"
    )

    return True, (
        f"El modelo {direccion} {abs(sesgo):.1f} clientes de "
        f"media ({abs(empleados):.2f} empleados) durante las "
        f"últimas {ventanas} ventanas, siempre en la misma "
        f"dirección. Eso ya saca decisiones de la banda."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MONITORIZACIÓN DEL MODELO")
    print("=" * 60)

    ruta = (
        PROCESSED_RESULTADOS_DIR
        / f"predicciones_{MODELO_PRODUCCION}.csv"
    )

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existen predicciones para monitorizar:\n{ruta}"
        )

    predicciones = pd.read_csv(ruta)

    predicciones = predicciones[
        predicciones["conjunto"] == "test"
    ]

    print()
    print(f"Modelo vigilado:   {MODELO_PRODUCCION}")
    print(f"Ventana:           {DIAS_VENTANA} días")

    print(
        f"Umbral de aviso:   {UMBRAL_SESGO:.1f} clientes "
        f"(medio empleado)"
    )

    # --------------------------------------------------------
    # SEGUIMIENTO
    # --------------------------------------------------------

    resumen = seguimiento(predicciones)

    print()
    print("SEGUIMIENTO POR VENTANAS")
    print("-" * 60)

    for _, fila in resumen.iterrows():

        if not fila["fiable"]:
            marca = "   (pocos días, no cuenta)"

        elif fila["aviso"]:
            marca = "  <-- AVISO"

        else:
            marca = ""

        print(
            f"  {fila['desde'].date()} -> "
            f"{fila['hasta'].date()}   "
            f"{int(fila['dias']):>3} días   "
            f"sesgo {fila['sesgo']:>+7.2f}   "
            f"({fila['empleados_de_sesgo']:>+5.2f} empleados)"
            f"{marca}"
        )

    # --------------------------------------------------------
    # MÉTRICAS GLOBALES
    # --------------------------------------------------------

    metricas = calcular_metricas(
        predicciones["n_clientes"],
        predicciones["prediccion"],
    )

    sesgo_global = float(
        (
            predicciones["n_clientes"]
            - predicciones["prediccion"]
        ).mean()
    )

    print()
    print("GLOBAL")
    print("-" * 60)

    print(f"  MAE:          {metricas['mae']:.2f} clientes")
    print(f"  Sesgo medio:  {sesgo_global:+.2f} clientes")

    print(
        f"                ({sesgo_global / RATIO_CLIENTES_EMPLEADO_OBJETIVO:+.2f} "
        f"empleados)"
    )

    # --------------------------------------------------------
    # DIAGNÓSTICO
    # --------------------------------------------------------

    hay_que_reentrenar, motivo = diagnostico(resumen)

    print()
    print("DIAGNÓSTICO")
    print("-" * 60)

    print(f"  {motivo}")

    print()

    if hay_que_reentrenar:

        print(
            "  ACCIÓN: reentrenar incorporando los datos "
            "recientes."
        )

        print()
        print("      python pipeline.py --desde separar")
        print()

        print(
            "  Reentrenar es la corrección adecuada: acerca el "
            "modelo al nivel actual"
        )

        print(
            "  sin extrapolar nada. Lo que NO hay que hacer es "
            "meter una tendencia"
        )

        print(
            "  temporal: está medido en "
            "src/analysis/correccion_tendencia.py y empeora"
        )

        print("  la decisión.")

    else:

        print("  ACCIÓN: ninguna. Seguir vigilando.")

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    resumen.to_csv(SALIDA, index=False, encoding="utf-8")

    print()
    print(f"Guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
