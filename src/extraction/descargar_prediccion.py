# src/extraction/descargar_prediccion.py

# ============================================================
# DESCARGAR PREDICCIÓN METEOROLÓGICA DE AEMET
# ============================================================

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import requests


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

# Permite ejecutar el script directamente desde:
# src/extraction/descargar_prediccion.py

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    AEMET_API_KEY,
    AEMET_ESTACION_PREDICCION,
    RAW_METEOROLOGIA_DIR,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_URL = "https://opendata.aemet.es"

ARCHIVO_FINAL = (
    RAW_METEOROLOGIA_DIR / "prediccion.json"
)

TIMEOUT = 60


# ============================================================
# SESIONES
# ============================================================

HEADERS = {
    "api_key": AEMET_API_KEY,
}


# ============================================================
# FUNCIONES
# ============================================================

def construir_url():
    """
    Construye la URL de AEMET para obtener la predicción
    meteorológica de la localidad configurada.

    AEMET devuelve inicialmente una respuesta que contiene
    una URL temporal en el campo 'datos'.
    """

    url = (
        f"{BASE_URL}/opendata/api/"
        f"prediccion/especifica/municipio/diaria/"
        f"{AEMET_ESTACION_PREDICCION}"
    )

    return url


def descargar_prediccion():
    """
    Realiza la petición a AEMET y descarga la predicción.

    La primera petición devuelve una URL temporal.
    Posteriormente se realiza una segunda petición para
    obtener los datos meteorológicos.
    """

    url = construir_url()

    print()
    print("------------------------------------------------------------")
    print("DESCARGANDO PREDICCIÓN METEOROLÓGICA")
    print("------------------------------------------------------------")
    print(f"Localidad: {AEMET_ESTACION_PREDICCION}")
    print()
    print("URL:")
    print(url)
    print()

    # --------------------------------------------------------
    # PRIMERA PETICIÓN
    # --------------------------------------------------------

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    print(f"HTTP: {response.status_code}")

    if response.status_code != 200:

        print(response.text)

        raise RuntimeError(
            f"Error al solicitar la predicción "
            f"a AEMET: HTTP {response.status_code}"
        )

    respuesta = response.json()

    # --------------------------------------------------------
    # COMPROBAR RESPUESTA DE AEMET
    # --------------------------------------------------------

    if respuesta.get("estado") != 200:

        raise RuntimeError(
            f"AEMET ha devuelto estado "
            f"{respuesta.get('estado')}: "
            f"{respuesta.get('descripcion')}"
        )

    url_datos = respuesta.get("datos")

    if not url_datos:

        raise RuntimeError(
            "AEMET no ha devuelto la URL de los datos."
        )

    print("URL de datos obtenida correctamente.")
    print("Descargando predicción...")

    # --------------------------------------------------------
    # SEGUNDA PETICIÓN
    # --------------------------------------------------------

    response_datos = requests.get(
        url_datos,
        timeout=TIMEOUT,
    )

    if response_datos.status_code != 200:

        raise RuntimeError(
            f"Error descargando los datos de predicción: "
            f"HTTP {response_datos.status_code}"
        )

    datos = response_datos.json()

    return datos


def guardar_prediccion(datos):
    """
    Guarda la predicción recibida de AEMET como JSON.
    """

    RAW_METEOROLOGIA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        ARCHIVO_FINAL,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            datos,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("Predicción guardada correctamente.")
    print(f"Archivo: {ARCHIVO_FINAL}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DESCARGA DE PREDICCIÓN AEMET")
    print("=" * 60)

    print()
    print(
        f"Localidad AEMET: "
        f"{AEMET_ESTACION_PREDICCION}"
    )

    print(
        f"Carpeta salida: "
        f"{RAW_METEOROLOGIA_DIR}"
    )

    print(
        f"Archivo salida: "
        f"{ARCHIVO_FINAL}"
    )

    # --------------------------------------------------------
    # COMPROBAR API KEY
    # --------------------------------------------------------

    if not AEMET_API_KEY:

        raise RuntimeError(
            "No se ha encontrado AEMET_API_KEY.\n"
            "Configúrala en el archivo .env "
            "o en src/config.py."
        )

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    manana = date.today() + timedelta(days=1)

    print()
    print(
        f"Predicción solicitada para: {manana}"
    )

    # --------------------------------------------------------
    # DESCARGAR
    # --------------------------------------------------------

    datos = descargar_prediccion()

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    guardar_prediccion(datos)

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        f"Fecha de predicción: {manana}"
    )

    print(
        f"Archivo: {ARCHIVO_FINAL}"
    )

    print()
    print(
        "Descarga de predicción finalizada."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()