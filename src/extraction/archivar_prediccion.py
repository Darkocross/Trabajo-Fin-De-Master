# src/extraction/archivar_prediccion.py

# ============================================================
# ARCHIVO HISTÓRICO DE PREVISIONES
# ============================================================
#
# AEMET publica la predicción a 7 días, pero no guarda las
# predicciones pasadas. Eso impide comparar lo que se dijo con
# lo que luego pasó, que es la única forma de medir de verdad
# cuánto se degrada el sistema al usar previsión en lugar de
# observación (ver src/analysis/robustez_prevision.py, que de
# momento lo simula).
#
# La solución es sencilla: guardar cada día la previsión que
# publica AEMET, con la fecha en la que se hizo. Al cabo de
# unos meses se tiene el conjunto de verificación.
#
# Uso recomendado: una tarea programada diaria que ejecute
#
#     python -m src.extraction.descargar_prediccion
#     python -m src.extraction.archivar_prediccion
#
# ============================================================

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    METEOROLOGIA_HISTORICA,
    PREDICCION_AEMET,
    RAW_METEOROLOGIA_DIR,
)


ARCHIVO_DIR = RAW_METEOROLOGIA_DIR / "archivo_predicciones"

VERIFICACION = (
    RAW_METEOROLOGIA_DIR / "verificacion_prevision.csv"
)


# ============================================================
# ARCHIVAR
# ============================================================

def archivar():
    """
    Guarda una copia de la previsión actual, identificada por
    la fecha en la que se elaboró.
    """

    if not PREDICCION_AEMET.exists():

        raise FileNotFoundError(
            f"No existe la previsión: {PREDICCION_AEMET}\n\n"
            "Descárgala antes con:\n"
            "    python -m src.extraction.descargar_prediccion"
        )

    with open(
        PREDICCION_AEMET,
        "r",
        encoding="utf-8",
    ) as archivo:

        datos = json.load(archivo)

    contenido = datos[0] if isinstance(datos, list) else datos

    elaborado = str(contenido.get("elaborado", ""))[:10]

    if not elaborado:
        elaborado = date.today().isoformat()

    ARCHIVO_DIR.mkdir(parents=True, exist_ok=True)

    destino = ARCHIVO_DIR / f"prediccion_{elaborado}.json"

    if destino.exists():

        print(
            f"La previsión del {elaborado} ya estaba "
            "archivada."
        )

        return destino, False

    shutil.copy2(PREDICCION_AEMET, destino)

    print(f"Previsión archivada: {destino.name}")

    return destino, True


# ============================================================
# EXTRAER LO PREVISTO
# ============================================================

def extraer_previsiones():
    """
    Recorre el archivo y construye una tabla con lo que se
    predijo para cada día y con cuánta antelación.
    """

    if not ARCHIVO_DIR.exists():
        return pd.DataFrame()

    filas = []

    for fichero in sorted(ARCHIVO_DIR.glob("prediccion_*.json")):

        with open(fichero, "r", encoding="utf-8") as archivo:

            datos = json.load(archivo)

        contenido = (
            datos[0] if isinstance(datos, list) else datos
        )

        elaborado = pd.Timestamp(
            str(contenido.get("elaborado", ""))[:10]
        )

        for dia in contenido.get(
            "prediccion", {}
        ).get("dia", []):

            temperatura = dia.get("temperatura", {})

            tmax = temperatura.get("maxima")
            tmin = temperatura.get("minima")

            if tmax is None or tmin is None:
                continue

            fecha = pd.Timestamp(dia["fecha"]).normalize()

            probabilidades = [
                elemento.get("value")
                for elemento in dia.get("probPrecipitacion", [])
                if elemento.get("periodo") == "00-24"
                and elemento.get("value") not in (None, "")
            ]

            filas.append({
                "fecha": fecha,
                "elaborado": elaborado,
                "antelacion_dias": (fecha - elaborado).days,
                "tmax_prevista": float(tmax),
                "tmin_prevista": float(tmin),
                "prob_lluvia_prevista": (
                    float(probabilidades[0])
                    if probabilidades
                    else 0.0
                ),
            })

    return pd.DataFrame(filas)


# ============================================================
# VERIFICAR
# ============================================================

def verificar(previsiones):
    """
    Cruza lo previsto con lo observado y calcula el error real
    de la previsión por antelación.
    """

    if previsiones.empty:
        return pd.DataFrame(), pd.DataFrame()

    if not METEOROLOGIA_HISTORICA.exists():
        return previsiones, pd.DataFrame()

    observado = pd.read_csv(
        METEOROLOGIA_HISTORICA,
        parse_dates=["fecha"],
    )[["fecha", "tmax", "tmin", "prec"]]

    cruce = previsiones.merge(
        observado,
        on="fecha",
        how="inner",
    )

    if cruce.empty:
        return cruce, pd.DataFrame()

    cruce["error_tmax"] = (
        cruce["tmax_prevista"] - cruce["tmax"]
    )

    cruce["error_tmin"] = (
        cruce["tmin_prevista"] - cruce["tmin"]
    )

    cruce["llovio"] = (cruce["prec"] > 0.5).astype(int)

    cruce["preveia_lluvia"] = (
        cruce["prob_lluvia_prevista"] >= 50
    ).astype(int)

    cruce["acierto_lluvia"] = (
        cruce["llovio"] == cruce["preveia_lluvia"]
    ).astype(int)

    resumen = (
        cruce
        .groupby("antelacion_dias")
        .agg(
            dias=("fecha", "size"),
            mae_tmax=("error_tmax", lambda x: x.abs().mean()),
            mae_tmin=("error_tmin", lambda x: x.abs().mean()),
            acierto_lluvia=("acierto_lluvia", "mean"),
        )
        .round(2)
    )

    return cruce, resumen


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ARCHIVO DE PREVISIONES METEOROLÓGICAS")
    print("=" * 60)

    print()

    archivar()

    previsiones = extraer_previsiones()

    print()
    print(
        f"Previsiones archivadas: "
        f"{len(list(ARCHIVO_DIR.glob('*.json')))} ficheros, "
        f"{len(previsiones)} días-previsión"
    )

    cruce, resumen = verificar(previsiones)

    if resumen.empty:

        print()
        print(
            "Todavía no hay días con previsión Y observación "
            "para comparar."
        )

        print()
        print(
            "Ejecuta este script a diario durante unas "
            "semanas. Cuando haya"
        )

        print(
            "suficientes días, aquí aparecerá el error real "
            "de AEMET por"
        )

        print(
            "antelación, y podrá sustituir a los supuestos de "
            "robustez_prevision.py."
        )

        return previsiones

    print()
    print("ERROR REAL DE LA PREVISIÓN")
    print("-" * 60)
    print(resumen.to_string())

    cruce.to_csv(
        VERIFICACION,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Guardado en: {VERIFICACION}")

    return cruce


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
