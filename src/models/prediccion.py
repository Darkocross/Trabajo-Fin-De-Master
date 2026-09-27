# src/models/prediccion.py

# ============================================================
# LÓGICA DE PREDICCIÓN
# ============================================================
#
# Módulo compartido entre:
#
#   - `src/models/prediccion_manana.py` (predicción diaria
#     automática con la previsión de AEMET).
#   - `app/app.py` (aplicación web).
#
# Ambos deben dar exactamente el mismo resultado para las
# mismas condiciones. Por eso la lógica vive aquí y no
# duplicada en cada uno.
#
# El módulo se encarga de:
#
#   1. Construir la fila de variables a partir de una fecha y
#      unas condiciones meteorológicas.
#   2. Aplicar el modelo de producción.
#   3. Devolver la predicción con su intervalo.
#   4. Traducir la predicción en una recomendación de plantilla.
#   5. Explicar de dónde sale el número.
#
# ============================================================

import sys
from datetime import date
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
    FECHA_ORIGEN,
    CALENDARIO,
    EMPLEADOS_MAXIMO,
    EMPLEADOS_MINIMO,
    GOLD_DATASET,
    RATIO_CLIENTES_EMPLEADO_OBJETIVO,
    RATIO_FALTA_PERSONAL,
    RATIO_PERSONAL_OCUPADO,
    ruta_modelo,
)

from src.models.entrenamiento import (
    aplicar_intervalo,
    cargar_metadatos,
)

from src.models.features import (
    VARIABLES,
    obtener_x,
    preparar_dataset,
)


# ============================================================
# UTILIDADES
# ============================================================

def texto(valor):
    """
    Convierte a texto un valor que puede venir vacío.

    Al leer el calendario desde CSV, las celdas vacías llegan
    como NaN (un float), no como cadena vacía. Sin esta
    normalización, cualquier operación de texto sobre ellas
    falla.
    """

    if valor is None:
        return ""

    if isinstance(valor, float) and pd.isna(valor):
        return ""

    texto_valor = str(valor).strip()

    if texto_valor.lower() in ("nan", "none", "<na>"):
        return ""

    return texto_valor


# ============================================================
# CARGA DEL MODELO
# ============================================================

def cargar_modelo_produccion():
    """
    Carga el modelo desplegado y sus metadatos.
    """

    import joblib

    ruta = ruta_modelo("produccion")

    if not ruta.exists():

        raise FileNotFoundError(
            "No existe el modelo de producción.\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py"
        )

    modelo = joblib.load(ruta)

    metadatos = cargar_metadatos("produccion")

    return modelo, metadatos


# ============================================================
# CALENDARIO
# ============================================================

def cargar_calendario():
    """
    Carga el calendario del proyecto.
    """

    if not CALENDARIO.exists():

        raise FileNotFoundError(
            f"No existe el calendario: {CALENDARIO}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py --hasta calendario"
        )

    return pd.read_csv(
        CALENDARIO,
        parse_dates=["fecha"],
    )


def obtener_dia_calendario(fecha, calendario=None):
    """
    Devuelve la fila del calendario correspondiente a una
    fecha.

    Si la fecha está fuera del periodo generado, se calculan
    las variables de calendario sobre la marcha con las mismas
    reglas, de modo que la aplicación nunca se queda sin
    respuesta.
    """

    fecha = pd.Timestamp(fecha).normalize()

    if calendario is None:
        calendario = cargar_calendario()

    fila = calendario[calendario["fecha"] == fecha]

    if not fila.empty:
        return fila.iloc[0].to_dict(), True

    # --- Fuera del calendario generado ---

    from src.transformation.generar_calendario import (
        generar_calendario,
    )

    año = int(fecha.year)

    generado = generar_calendario(año, año)

    generado["fecha"] = pd.to_datetime(generado["fecha"])

    fila = generado[generado["fecha"] == fecha]

    if fila.empty:

        raise ValueError(
            f"No se ha podido construir el calendario para "
            f"{fecha.date()}."
        )

    return fila.iloc[0].to_dict(), False


# ============================================================
# CONSTRUCCIÓN DE LA FILA DE ENTRADA
# ============================================================

def construir_entrada(
    fecha,
    tmax,
    tmin,
    prec,
    tmed=None,
    calendario=None,
    es_festivo=None,
):
    """
    Construye la fila de variables que espera el modelo.

    Las variables de calendario se toman del calendario del
    proyecto, para que la predicción use exactamente la misma
    definición de festivo, vacaciones o Semana Santa con la
    que se entrenó el modelo.

    `es_festivo` permite forzar el valor desde la aplicación,
    por si el usuario sabe de un festivo local que el
    calendario no recoge.
    """

    fecha = pd.Timestamp(fecha).normalize()

    dia, en_calendario = obtener_dia_calendario(
        fecha,
        calendario,
    )

    # --------------------------------------------------------
    # TEMPERATURA MEDIA
    # --------------------------------------------------------
    #
    # AEMET calcula la media como el punto medio entre la
    # máxima y la mínima. Si no nos la dan, la reproducimos
    # igual.

    if tmed is None:
        tmed = (float(tmax) + float(tmin)) / 2

    # --------------------------------------------------------
    # FESTIVO
    # --------------------------------------------------------

    if es_festivo is None:
        es_festivo = int(dia["es_festivo"])

    else:
        es_festivo = int(bool(es_festivo))

    fila = {
        "fecha": fecha,

        "dia_semana": int(dia["dia_semana_num"]),
        "mes": int(dia["mes"]),
        "fin_de_semana": int(dia["fin_de_semana"]),

        "tipo_vacaciones": tipo_vacaciones_desde_periodo(
            dia.get("periodo_vacacional")
        ),

        "es_festivo": es_festivo,

        # Tendencia: días transcurridos desde el origen fijo
        # del proyecto. Es lo que permite al modelo saber que
        # el negocio ha crecido desde 2022.
        "indice_temporal": int(
            (fecha.date() - FECHA_ORIGEN).days
        ),

        # El calendario publica estas dos porque son hechos
        # del almanaque. Entran en el modelo solo si
        # USAR_VISPERAS_Y_PUENTES está encendido, pero se
        # rellenan siempre para no depender del interruptor.
        "es_vispera_festivo": int(
            dia.get("es_vispera_festivo", 0) or 0
        ),
        "es_puente": int(
            dia.get("es_puente", 0) or 0
        ),

        "tmed": float(tmed),
        "tmax": float(tmax),
        "tmin": float(tmin),
        "prec": float(prec),
    }

    datos = pd.DataFrame([fila])

    datos = preparar_dataset(datos, exigir_objetivo=False)

    return datos, dia, en_calendario


def tipo_vacaciones_desde_periodo(periodo):
    """
    Misma traducción que usa la capa gold.
    """

    from src.transformation.crear_gold import (
        obtener_tipo_vacaciones,
    )

    return obtener_tipo_vacaciones(periodo)


# ============================================================
# RECOMENDACIÓN DE PLANTILLA
# ============================================================

def recomendar_empleados(clientes):
    """
    Traduce el número de clientes previsto en una plantilla
    recomendada.

    Los umbrales no son inventados: son los mismos con los que
    se etiqueta la variable `nota_faena` en los datos de
    actividad.

        más de 24 clientes por empleado -> falta de personal
        entre 15 y 24                   -> personal ocupado
        menos de 15                     -> personal ocioso

    Se apunta al centro de la banda cómoda para dejar margen
    en ambas direcciones.
    """

    clientes = max(float(clientes), 0.0)

    empleados = clientes / RATIO_CLIENTES_EMPLEADO_OBJETIVO

    empleados = int(np.ceil(empleados))

    return int(
        np.clip(
            empleados,
            EMPLEADOS_MINIMO,
            EMPLEADOS_MAXIMO,
        )
    )


def evaluar_plantilla(clientes, empleados):
    """
    Dice cómo quedaría la jornada con una plantilla concreta.
    """

    empleados = max(int(empleados), 1)

    ratio = float(clientes) / empleados

    if ratio >= RATIO_FALTA_PERSONAL:

        return (
            "riesgo",
            "Faltaría personal",
            f"{ratio:.0f} clientes por empleado",
        )

    if ratio >= RATIO_PERSONAL_OCUPADO:

        return (
            "ajustado",
            "Personal ocupado",
            f"{ratio:.0f} clientes por empleado",
        )

    return (
        "holgado",
        "Personal holgado",
        f"{ratio:.0f} clientes por empleado",
    )


# ============================================================
# PREDICCIÓN
# ============================================================

def predecir(
    fecha,
    tmax,
    tmin,
    prec,
    tmed=None,
    modelo=None,
    metadatos=None,
    calendario=None,
    es_festivo=None,
):
    """
    Devuelve la predicción completa para un día:

        - número de clientes estimado
        - intervalo de predicción al 80 %
        - plantilla recomendada
        - contexto del día (festivo, vacaciones, cierre...)

    Es la función que usan tanto la aplicación como la
    predicción diaria automática.
    """

    if modelo is None or metadatos is None:
        modelo, metadatos = cargar_modelo_produccion()

    datos, dia, en_calendario = construir_entrada(
        fecha=fecha,
        tmax=tmax,
        tmin=tmin,
        prec=prec,
        tmed=tmed,
        calendario=calendario,
        es_festivo=es_festivo,
    )

    X = obtener_x(datos)

    prediccion = float(
        np.clip(
            modelo.predict(X)[0],
            0,
            None,
        )
    )

    # --------------------------------------------------------
    # INTERVALO
    # --------------------------------------------------------

    # Se prefiere el intervalo que escala con la predicción:
    # un martes flojo y un sábado lleno no tienen la misma
    # incertidumbre, y darles la misma anchura sobra en el
    # primero y falta en el segundo.
    #
    # Si el modelo es antiguo y no lo trae, se usa el fijo.

    escalado = metadatos.get("intervalo_80_escalado")

    if escalado:

        inferior = aplicar_intervalo(
            prediccion,
            escalado["inferior"],
        )

        superior = aplicar_intervalo(
            prediccion,
            escalado["superior"],
        )

    else:

        intervalo = metadatos.get("intervalo_80", {})

        inferior = (
            prediccion + float(intervalo.get("inferior", 0.0))
        )

        superior = (
            prediccion + float(intervalo.get("superior", 0.0))
        )

    inferior = max(0.0, inferior)

    superior = max(superior, inferior)

    # --------------------------------------------------------
    # PLANTILLA
    # --------------------------------------------------------

    empleados = recomendar_empleados(prediccion)

    # --------------------------------------------------------
    # CIERRE
    # --------------------------------------------------------

    cerrado = int(dia.get("restaurante_cerrado", 0)) == 1

    return {
        "fecha": pd.Timestamp(fecha).normalize(),

        "prediccion": prediccion,
        "intervalo_inferior": inferior,
        "intervalo_superior": superior,

        "empleados_recomendados": empleados,

        "cerrado": cerrado,
        "motivo_cierre": texto(dia.get("motivo_cierre")),

        "nombre_dia": texto(dia.get("dia_semana")),
        "es_festivo": int(dia.get("es_festivo", 0)),
        "nombre_festivo": texto(dia.get("nombre_festivo")),
        "periodo_vacacional": texto(
            dia.get("periodo_vacacional")
        ),

        "tmed": float(datos["tmed"].iloc[0]),
        "tmax": float(tmax),
        "tmin": float(tmin),
        "prec": float(prec),

        "en_calendario": en_calendario,
        "variables": datos[VARIABLES].iloc[0].to_dict(),
    }


# ============================================================
# CONTEXTO HISTÓRICO
# ============================================================

def contexto_historico(fecha, gold=None):
    """
    Devuelve la referencia histórica de ese día de la semana,
    para que el usuario pueda comparar la predicción con lo
    que suele pasar.

    Una cifra sola no dice nada. "145 clientes" solo significa
    algo si sabes que un sábado normal vienen 120.
    """

    if gold is None:

        if not GOLD_DATASET.exists():
            return None

        gold = pd.read_csv(
            GOLD_DATASET,
            parse_dates=["fecha"],
        )

    fecha = pd.Timestamp(fecha).normalize()

    dia_semana = fecha.weekday()

    mismos_dias = gold[gold["dia_semana"] == dia_semana]

    if mismos_dias.empty:
        return None

    # Mismo día de la semana y mismo mes: la comparación más
    # justa posible.
    mismo_mes = mismos_dias[
        mismos_dias["mes"] == fecha.month
    ]

    return {
        "media_dia_semana": float(
            mismos_dias["n_clientes"].mean()
        ),
        "media_dia_semana_mes": (
            float(mismo_mes["n_clientes"].mean())
            if not mismo_mes.empty
            else None
        ),
        "registros_dia_semana_mes": int(len(mismo_mes)),
        "minimo": float(mismos_dias["n_clientes"].min()),
        "maximo": float(mismos_dias["n_clientes"].max()),
    }


# ============================================================
# EXPLICACIÓN
# ============================================================

def explicar(resultado, historico=None):
    """
    Construye una explicación en lenguaje natural de por qué
    sale ese número.

    No usa IA generativa: son reglas sobre los valores reales
    de las variables que ha recibido el modelo. Así la
    explicación nunca puede inventar una causa que no esté en
    los datos.
    """

    motivos = []

    nombre_dia = str(resultado.get("nombre_dia", "")).lower()

    # --- Día de la semana ---

    if nombre_dia in ("sábado", "domingo", "viernes"):

        motivos.append(
            f"Es {nombre_dia}, el tramo de mayor afluencia "
            "de la semana."
        )

    elif nombre_dia in ("lunes", "martes"):

        motivos.append(
            f"Es {nombre_dia}, uno de los días más flojos."
        )

    # --- Festivo ---

    if resultado.get("es_festivo"):

        festivo = resultado.get("nombre_festivo") or "festivo"

        motivos.append(f"Es festivo ({festivo}).")

    # --- Vacaciones ---

    periodo = resultado.get("periodo_vacacional") or ""

    if periodo and periodo.lower() != "ninguna":

        motivos.append(f"Cae en {periodo.lower()}.")

    # --- Meteorología ---

    tmed = resultado.get("tmed", 0)
    tmax = resultado.get("tmax", 0)
    tmin = resultado.get("tmin", 0)
    prec = resultado.get("prec", 0)

    if prec >= 5:

        motivos.append(
            f"Se esperan {prec:.0f} mm de lluvia, que suelen "
            "reducir la afluencia."
        )

    elif prec > 0:

        motivos.append(
            f"Se esperan lluvias débiles ({prec:.1f} mm)."
        )

    if tmax >= 32:

        motivos.append(
            f"Calor intenso ({tmax:.0f} °C de máxima), que "
            "penaliza la terraza al mediodía."
        )

    elif tmin <= 5:

        motivos.append(
            f"Frío intenso ({tmin:.0f} °C de mínima)."
        )

    elif 16 <= tmed <= 26:

        motivos.append(
            f"Temperatura agradable ({tmed:.0f} °C de media)."
        )

    # --- Comparación histórica ---

    if historico and historico.get("media_dia_semana_mes"):

        media = historico["media_dia_semana_mes"]

        diferencia = resultado["prediccion"] - media

        if abs(diferencia) < 5:

            motivos.append(
                "La previsión está en línea con la media "
                "histórica de este día."
            )

        elif diferencia > 0:

            motivos.append(
                f"La previsión está {diferencia:.0f} clientes "
                "por encima de la media histórica de este día."
            )

        else:

            motivos.append(
                f"La previsión está {abs(diferencia):.0f} "
                "clientes por debajo de la media histórica de "
                "este día."
            )

    return motivos
