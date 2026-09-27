# src/extraction/descargar_meteorologia.py

# ============================================================
# DESCARGAR METEOROLOGÍA DE AEMET
# ============================================================
#
# Descarga el histórico de valores climatológicos diarios de
# la estación 3182Y y lo acumula en data/raw/meteorologia.
#
# ------------------------------------------------------------
# REGLA DE DISEÑO: LA CAPA RAW NO INTERPRETA
# ------------------------------------------------------------
#
# Este módulo guarda lo que AEMET envía. No convierte a número,
# no rellena y no decide qué significa un valor raro. Esas son
# decisiones de análisis y viven en la capa de transformación,
# donde se pueden documentar y discutir.
#
# El motivo no es purismo. AEMET no solo devuelve números:
#
#   "Ip"     Precipitación inapreciable (por debajo de 0,1 mm).
#            SÍ llovió. Es una medición, no un hueco.
#   "Acum"   El dato está acumulado en otra fecha.
#   ""       No hay medición.
#
# Una versión anterior hacía `pd.to_numeric(errors="coerce")`
# aquí, antes de guardar. Eso convertía las tres situaciones en
# la misma celda vacía, sin avisar, y el CSV de la capa raw ya
# no permitía distinguirlas. La capa gold rellenaba todo con 0
# y lo marcaba como imputado.
#
# El daño real se vio al revisar los huecos: 33 días seguidos
# sin precipitación (19/10/2022 - 20/11/2022) con todas las
# temperaturas correctas. Eso no es lluvia inapreciable, es el
# pluviómetro averiado durante un mes. Con la conversión hecha
# aquí, esa diferencia era invisible.
#
# ------------------------------------------------------------
# REGLA DE DISEÑO: LOS HUECOS SE VUELVEN A PEDIR
# ------------------------------------------------------------
#
# AEMET publica con retraso, y a veces completa días semanas
# después. Una descarga que solo pide "desde el último día que
# tengo" convierte cualquier hueco interior en permanente.
#
# Este módulo calcula qué días faltan o están incompletos en el
# archivo acumulado y los vuelve a pedir, además de los días
# nuevos. Si AEMET ya los ha publicado, se recuperan datos
# reales en lugar de conservar una imputación.
#
# ============================================================

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

# Permite ejecutar el script directamente desde:
# src/extraction/descargar_meteorologia.py
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    AEMET_API_KEY,
    AEMET_ESTACION_HISTORICO,
    RAW_METEOROLOGIA_DIR
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_URL = "https://opendata.aemet.es"


ARCHIVO_FINAL = RAW_METEOROLOGIA_DIR / "meteorologia.csv"

TIMEOUT = 60


# ============================================================
# CÓDIGOS DE AEMET
# ============================================================
#
# Valores no numéricos que AEMET devuelve en las columnas de
# medida. Se conservan tal cual en la capa raw; los interpreta
# src/transformation/crear_gold.py.

MARCAS_AEMET = {
    "Ip": "Precipitación inapreciable (< 0,1 mm)",
    "Acum": "Acumulado en otra fecha",
    "Varias": "Varias observaciones en el día",
}


# Columnas de medida que se conservan.

COLUMNAS_MEDIDA = [
    "tmed",
    "tmin",
    "tmax",
    "prec",
]


# ============================================================
# SESIONES
# ============================================================

HEADERS = {
    "api_key": AEMET_API_KEY,
}


# ============================================================
# FUNCIONES
# ============================================================

def construir_url(fecha_inicio, fecha_fin):
    """
    Construye la URL de AEMET sin codificar los ':' de las fechas.
    """

    fecha_inicio_str = (
        f"{fecha_inicio.isoformat()}T00:00:00UTC"
    )

    fecha_fin_str = (
        f"{fecha_fin.isoformat()}T23:59:59UTC"
    )

    url = (
        f"{BASE_URL}/opendata/api/"
        f"valores/climatologicos/diarios/datos/"
        f"fechaini/{fecha_inicio_str}/"
        f"fechafin/{fecha_fin_str}/"
        f"estacion/{AEMET_ESTACION_HISTORICO}"
    )

    return url


def descargar_periodo(fecha_inicio, fecha_fin):
    """
    Descarga un periodo de datos de AEMET.

    La primera petición devuelve una URL temporal en el campo
    'datos'. Después hacemos una segunda petición para obtener
    los datos meteorológicos.
    """

    url = construir_url(fecha_inicio, fecha_fin)

    print()
    print("------------------------------------------------------------")
    print("DESCARGANDO PERIODO")
    print("------------------------------------------------------------")
    print(f"Desde:     {fecha_inicio}")
    print(f"Hasta:     {fecha_fin}")
    print(f"Estación:  {AEMET_ESTACION_HISTORICO}")
    print()
    print("URL:")
    print(url)
    print()

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    print(f"HTTP: {response.status_code}")

    if response.status_code != 200:
        print(response.text)

        raise RuntimeError(
            f"Error al solicitar AEMET: "
            f"HTTP {response.status_code}"
        )

    respuesta = response.json()

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
    print("Descargando datos...")

    response_datos = requests.get(
        url_datos,
        timeout=TIMEOUT,
    )

    if response_datos.status_code != 200:
        raise RuntimeError(
            f"Error descargando los datos: "
            f"HTTP {response_datos.status_code}"
        )

    datos = response_datos.json()

    if not isinstance(datos, list):
        raise RuntimeError(
            "El formato recibido de AEMET no es una lista."
        )

    print(f"Registros descargados: {len(datos)}")

    if len(datos) == 0:
        return pd.DataFrame()

    return pd.DataFrame(datos)


def normalizar_datos(df, verbose=True):
    """
    Normaliza el formato SIN interpretar el contenido.

    Lo único que se toca de las columnas de medida es la coma
    decimal, que se convierte en punto para que el fichero sea
    legible desde cualquier configuración regional. Un valor
    como "Ip" se guarda como "Ip".

    Los valores no numéricos se cuentan y se avisan, porque un
    código nuevo de AEMET que nadie ha visto es exactamente el
    tipo de cosa que conviene descubrir aquí y no tres capas
    más abajo.
    """

    if df.empty:
        return df

    df = df.copy()

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(
            df["fecha"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # COLUMNAS DE MEDIDA
    # --------------------------------------------------------

    marcas_encontradas = {}

    for columna in COLUMNAS_MEDIDA:

        if columna not in df.columns:
            continue

        valores = df[columna]

        # Solo la coma decimal. Nada más.

        texto = (
            valores
            .astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        texto = texto.mask(texto == "", pd.NA)

        df[columna] = texto

        # --- Inventario de valores no numéricos ---

        numerico = pd.to_numeric(texto, errors="coerce")

        no_numericos = texto[
            texto.notna() & numerico.isna()
        ]

        for marca, cuantos in no_numericos.value_counts().items():

            clave = (columna, str(marca))

            marcas_encontradas[clave] = (
                marcas_encontradas.get(clave, 0) + int(cuantos)
            )

    if verbose and marcas_encontradas:

        print()
        print("CÓDIGOS DE AEMET ENCONTRADOS (se conservan)")
        print("-" * 60)

        for (columna, marca), cuantos in sorted(
            marcas_encontradas.items()
        ):

            significado = MARCAS_AEMET.get(
                marca,
                "CÓDIGO DESCONOCIDO - revisar",
            )

            print(
                f"  {columna:<6} {marca:<8} "
                f"{cuantos:>4} días   {significado}"
            )

    return df


# ============================================================
# DETECCIÓN DE HUECOS
# ============================================================

def fechas_con_hueco(df):
    """
    Devuelve las fechas del archivo acumulado que conviene
    volver a pedir:

        - Las que no están en el fichero.
        - Las que están pero les falta alguna medida.

    Un código de AEMET ("Ip", "Acum") NO es un hueco: es una
    medición. Solo cuenta como hueco la celda vacía.
    """

    if df.empty:
        return []

    fechas = pd.to_datetime(df["fecha"]).dropna()

    if fechas.empty:
        return []

    # --- Fechas que no existen en el fichero ---

    esperadas = pd.date_range(
        fechas.min(),
        fechas.max(),
        freq="D",
    )

    ausentes = list(esperadas.difference(fechas))

    # --- Fechas presentes pero incompletas ---

    presentes = df.copy()
    presentes["fecha"] = pd.to_datetime(presentes["fecha"])

    columnas = [
        columna
        for columna in COLUMNAS_MEDIDA
        if columna in presentes.columns
    ]

    if columnas:

        incompletas = presentes[
            presentes[columnas].isna().any(axis=1)
        ]["fecha"].tolist()

    else:
        incompletas = []

    todas = sorted(set(ausentes) | set(incompletas))

    return [fecha.date() for fecha in todas]


def agrupar_en_bloques(fechas, margen_dias=2):
    """
    Agrupa fechas sueltas en bloques contiguos, con un margen a
    cada lado.

    Pedir a AEMET tres días sueltos cuesta tres peticiones;
    pedir el bloque que los contiene, una. El margen además
    recoge días vecinos que puedan haberse corregido.
    """

    if not fechas:
        return []

    fechas = sorted(fechas)

    bloques = []

    inicio = fechas[0]
    anterior = fechas[0]

    for fecha in fechas[1:]:

        # Si el salto es pequeño, sigue siendo el mismo bloque.

        if (fecha - anterior).days <= (2 * margen_dias + 1):
            anterior = fecha
            continue

        bloques.append((inicio, anterior))

        inicio = fecha
        anterior = fecha

    bloques.append((inicio, anterior))

    return [
        (
            inicio - timedelta(days=margen_dias),
            fin + timedelta(days=margen_dias),
        )
        for inicio, fin in bloques
    ]


def obtener_periodos(fecha_inicio, fecha_fin):
    """
    Genera bloques:

        1 enero -> 1 julio
        1 julio -> 31 diciembre

    para cada año.

    Se adaptan automáticamente a las fechas solicitadas.
    """

    periodos = []

    año = fecha_inicio.year

    while año <= fecha_fin.year:

        # ----------------------------------------------------
        # PRIMER BLOQUE
        # ----------------------------------------------------

        inicio = date(año, 1, 1)
        fin = date(año, 7, 1)

        inicio = max(inicio, fecha_inicio)
        fin = min(fin, fecha_fin)

        if inicio <= fin:
            periodos.append((inicio, fin))

        # ----------------------------------------------------
        # SEGUNDO BLOQUE
        # ----------------------------------------------------

        inicio = date(año, 7, 1)
        fin = date(año, 12, 31)

        inicio = max(inicio, fecha_inicio)
        fin = min(fin, fecha_fin)

        if inicio <= fin:
            periodos.append((inicio, fin))

        año += 1

    return periodos


def cargar_datos_existentes():
    """
    Carga el archivo acumulado si ya existe.
    """

    if not ARCHIVO_FINAL.exists():
        print("No existe todavía el archivo meteorológico.")
        return pd.DataFrame()

    print()
    print("Archivo meteorológico existente encontrado:")
    print(ARCHIVO_FINAL)

    try:

        df = pd.read_csv(
            ARCHIVO_FINAL,
            encoding="utf-8",
        )

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(
                df["fecha"],
                errors="coerce",
            )

        print(
            f"Registros existentes: {len(df)}"
        )

        if not df.empty:
            print(
                f"Fecha inicial: "
                f"{df['fecha'].min().date()}"
            )

            print(
                f"Fecha final:   "
                f"{df['fecha'].max().date()}"
            )

        return df

    except Exception as e:

        raise RuntimeError(
            f"No se pudo leer {ARCHIVO_FINAL}: {e}"
        )


def guardar_datos(df):
    """
    Guarda el dataset meteorológico acumulado.
    """

    RAW_METEOROLOGIA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = df.sort_values("fecha")

    df = df.drop_duplicates(
        subset=["fecha"],
        keep="last",
    )

    df.to_csv(
        ARCHIVO_FINAL,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Archivo guardado correctamente.")
    print(f"Archivo: {ARCHIVO_FINAL}")
    print(f"Registros: {len(df)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DESCARGA METEOROLÓGICA AEMET")
    print("=" * 60)

    print()
    print(f"Estación AEMET: {AEMET_ESTACION_HISTORICO}")
    print(f"Carpeta salida: {RAW_METEOROLOGIA_DIR}")

    # --------------------------------------------------------
    # COMPROBAR API KEY
    # --------------------------------------------------------

    if not AEMET_API_KEY:

        raise RuntimeError(
            "No se ha encontrado AEMET_API_KEY.\n"
            "Configúrala en el archivo .env o en src/config.py."
        )

    # --------------------------------------------------------
    # FECHAS
    # --------------------------------------------------------

    fecha_inicial_proyecto = date(2022, 1, 1)
    fecha_actual = date.today()

    # --------------------------------------------------------
    # CARGAR DATOS EXISTENTES
    # --------------------------------------------------------

    df_existente = cargar_datos_existentes()

    # --------------------------------------------------------
    # DETERMINAR DESDE CUÁNDO DESCARGAR
    # --------------------------------------------------------

    periodos = []

    if df_existente.empty:

        print()
        print("Primera ejecución.")

        periodos = obtener_periodos(
            fecha_inicial_proyecto,
            fecha_actual,
        )

    else:

        ultima_fecha = (
            df_existente["fecha"]
            .dropna()
            .max()
            .date()
        )

        print()
        print(
            f"Última fecha disponible: {ultima_fecha}"
        )

        # ----------------------------------------------------
        # 1. DÍAS NUEVOS
        # ----------------------------------------------------

        fecha_inicio = ultima_fecha + timedelta(days=1)

        if fecha_inicio <= fecha_actual:

            periodos += obtener_periodos(
                fecha_inicio,
                fecha_actual,
            )

            print(
                f"Días nuevos por descargar: "
                f"{fecha_inicio} -> {fecha_actual}"
            )

        else:
            print("No hay días nuevos que descargar.")

        # ----------------------------------------------------
        # 2. HUECOS DEL HISTÓRICO
        # ----------------------------------------------------
        #
        # AEMET publica con retraso y completa días semanas
        # después. Si no se vuelven a pedir, el hueco es
        # permanente y acaba relleno por imputación.

        huecos = fechas_con_hueco(df_existente)

        if huecos:

            bloques = agrupar_en_bloques(huecos)

            print()
            print(
                f"Huecos en el histórico: {len(huecos)} días "
                f"en {len(bloques)} bloques"
            )

            for inicio, fin in bloques:

                # Nunca por delante del inicio del proyecto ni
                # por detrás de hoy.

                inicio = max(inicio, fecha_inicial_proyecto)
                fin = min(fin, fecha_actual)

                if inicio > fin:
                    continue

                dias = (fin - inicio).days + 1

                print(f"  {inicio} -> {fin}  ({dias} días)")

                periodos += obtener_periodos(inicio, fin)

        else:
            print("No hay huecos en el histórico.")

    # --------------------------------------------------------
    # COMPROBAR SI HAY ALGO QUE HACER
    # --------------------------------------------------------

    if not periodos:

        print()
        print(
            "Los datos meteorológicos ya están completos "
            "y actualizados."
        )

        return

    print()
    print(
        f"Periodos a descargar: {len(periodos)}"
    )

    # --------------------------------------------------------
    # DESCARGAR
    # --------------------------------------------------------

    nuevos_datos = []

    for inicio, fin in periodos:

        try:

            df_periodo = descargar_periodo(
                inicio,
                fin,
            )

            if not df_periodo.empty:

                nuevos_datos.append(
                    df_periodo
                )

        except Exception as e:

            print()
            print(
                f"ERROR en el periodo "
                f"{inicio} -> {fin}"
            )
            print(str(e))

            # No detenemos toda la descarga.
            # Continuamos con el siguiente periodo.
            continue

    # --------------------------------------------------------
    # COMPROBAR RESULTADOS
    # --------------------------------------------------------

    if not nuevos_datos:

        print()
        print(
            "No se han obtenido datos nuevos."
        )

        return

    # --------------------------------------------------------
    # UNIR DESCARGAS
    # --------------------------------------------------------

    df_nuevo = pd.concat(
        nuevos_datos,
        ignore_index=True,
    )

    df_nuevo = normalizar_datos(
        df_nuevo
    )

    print()
    print(
        f"Nuevos registros descargados: "
        f"{len(df_nuevo)}"
    )

    # --------------------------------------------------------
    # UNIR CON HISTÓRICO
    # --------------------------------------------------------

    if df_existente.empty:

        df_final = df_nuevo

    else:

        df_final = pd.concat(
            [
                df_existente,
                df_nuevo,
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # NORMALIZAR TODO
    # --------------------------------------------------------

    df_final = normalizar_datos(
        df_final
    )

    # --------------------------------------------------------
    # ELIMINAR DUPLICADOS
    # --------------------------------------------------------

    df_final = df_final.drop_duplicates(
        subset=["fecha"],
        keep="last",
    )

    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    df_final = df_final.sort_values(
        "fecha"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    guardar_datos(
        df_final
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        f"Total registros: {len(df_final)}"
    )

    print(
        f"Fecha inicial: "
        f"{df_final['fecha'].min().date()}"
    )

    print(
        f"Fecha final:   "
        f"{df_final['fecha'].max().date()}"
    )

    # --------------------------------------------------------
    # HUECOS QUE SIGUEN ABIERTOS
    # --------------------------------------------------------

    huecos_finales = fechas_con_hueco(df_final)

    if df_existente.empty:
        recuperados = None

    else:
        recuperados = (
            len(fechas_con_hueco(df_existente))
            - len(huecos_finales)
        )

    print()

    if recuperados:
        print(f"Huecos recuperados en esta descarga: {recuperados}")

    if huecos_finales:

        print(
            f"Huecos que AEMET sigue sin publicar: "
            f"{len(huecos_finales)} días"
        )

        print(
            "Los rellena por imputación "
            "src/transformation/crear_gold.py, "
            "marcados en la capa gold."
        )

    else:
        print("El histórico no tiene huecos.")

    print()
    print(
        "Descarga finalizada."
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()