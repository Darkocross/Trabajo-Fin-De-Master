# src/transformation/generar_calendario.py

# ============================================================
# GENERACIÓN DEL CALENDARIO
# ============================================================
#
# Construye el calendario diario del restaurante para todo el
# periodo del proyecto.
#
# El calendario es la única fuente de verdad sobre:
#
#   - Festivos (nacionales, autonómicos y locales).
#   - Periodos vacacionales.
#   - Días de cierre del restaurante.
#
# ------------------------------------------------------------
# REGLA DE DISEÑO: HECHOS, NO HIPÓTESIS
# ------------------------------------------------------------
#
# Una columna solo entra en el calendario si supera esta prueba:
#
#   ¿Puede alguien comprobarla contra el BOE, el convenio o el
#   propio almanaque, sin tener que opinar?
#
# "Es festivo", "es víspera de festivo", "cae en fin de semana"
# o "el restaurante cierra" la superan. "Es un mes de alta
# demanda" no: eso es una hipótesis sobre el comportamiento del
# negocio, y las hipótesis se contrastan en el modelado.
#
# Esta regla corrigió un problema real. El calendario emitía
# es_mayo, es_junio, es_julio y sus interacciones con el fin de
# semana. Esos tres meses no salieron de los datos: salieron de
# los tres meses que el generador sintético amplifica. Es fuga
# conceptual — el calendario estaba copiando el proceso
# generador en lugar de describir el almanaque — y además tres
# meses elegidos a dedo, cuando la ratio findes/laborables más
# alta de la serie está en noviembre (1,955) y abril (1,913),
# por encima de mayo (1,839).
#
# Las interacciones no desaparecen del proyecto: se construyen
# en src/models/features.py, de forma sistemática (los doce
# meses, no tres) y sobre datos de entrenamiento, que es donde
# se pueden medir y descartar si no aportan.
#
# ------------------------------------------------------------
#
# Los festivos NO se escriben a mano año por año: se calculan a
# partir de reglas. La Semana Santa se obtiene con el algoritmo
# de Butcher-Meeus, de modo que el calendario se puede extender
# a cualquier año sin tocar el código.
#
# Cada festivo lleva su ámbito (nacional, autonómico o local).
# Esto deja la puerta abierta a datos reales: si más adelante se
# incorpora otro municipio, basta con añadir sus fiestas locales
# y el resto del calendario sigue siendo válido.
#
# Localización del proyecto:
#
#   Arganda del Rey (Comunidad de Madrid)
#
# Se ha elegido esta localidad porque es la que corresponde a
# los datos meteorológicos reales utilizados en el proyecto
# (estación AEMET 3182Y, Arganda del Rey) y al municipio de
# predicción (código INE 28014).
#
# Fuentes de los festivos:
#
#   - Calendario laboral de la Comunidad de Madrid.
#   - Fiestas locales de Arganda del Rey (Virgen de la Soledad,
#     celebradas en septiembre).
#
# ============================================================

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

# Permite ejecutar el script directamente desde:
# src/transformation/generar_calendario.py

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    AÑO_FIN,
    AÑO_INICIO,
    CALENDARIO,
    DIAS_DESCANSO,
    MES_CIERRE_VACACIONES,
    PROCESSED_CALENDARIO_DIR,
)


# ============================================================
# NOMBRES DE LOS DÍAS
# ============================================================

NOMBRES_DIAS = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]


# ============================================================
# CÁLCULO DE LA PASCUA
# ============================================================

def domingo_de_pascua(año):
    """
    Calcula el Domingo de Resurrección mediante el algoritmo
    de Butcher-Meeus (calendario gregoriano).

    A partir de esta fecha se obtienen el Jueves Santo, el
    Viernes Santo y el periodo vacacional de Semana Santa.
    """

    a = año % 19
    b = año // 100
    c = año % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3

    h = (19 * a + b - d - g + 15) % 30

    i = c // 4
    k = c % 4

    l = (32 + 2 * e + 2 * i - h - k) % 7

    m = (a + 11 * h + 22 * l) // 451

    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1

    return date(año, mes, dia)


def jueves_santo(año):
    """
    Jueves Santo: tres días antes del Domingo de Pascua.
    """

    return domingo_de_pascua(año) - timedelta(days=3)


def viernes_santo(año):
    """
    Viernes Santo: dos días antes del Domingo de Pascua.
    """

    return domingo_de_pascua(año) - timedelta(days=2)


# ============================================================
# TRASLADO DE FESTIVOS EN DOMINGO
# ============================================================

def trasladar_si_domingo(fecha):
    """
    Cuando un festivo nacional de fecha fija cae en domingo,
    la Comunidad de Madrid suele trasladarlo al lunes
    siguiente.

    Se aplica a:

        1 de noviembre (Todos los Santos)
        6 de diciembre (Día de la Constitución)
    """

    if fecha.weekday() == 6:
        return fecha + timedelta(days=1)

    return fecha


# ============================================================
# FESTIVOS DEL AÑO
# ============================================================

def festivos_del_año(año):
    """
    Devuelve un diccionario {fecha: (nombre, ámbito)} con todos
    los festivos aplicables a Arganda del Rey en un año dado.

    Se combinan tres niveles, y el ámbito se conserva:

        1. Festivos nacionales    -> "nacional"
        2. Festivo de Madrid      -> "autonomico"
        3. Fiestas de Arganda     -> "local"

    Guardar el ámbito permite reutilizar el calendario en otro
    municipio cambiando solo el bloque local.
    """

    festivos = {}

    # --------------------------------------------------------
    # 1. FESTIVOS NACIONALES DE FECHA FIJA
    # --------------------------------------------------------

    fijos = [
        (date(año, 1, 1), "Año Nuevo"),
        (date(año, 1, 6), "Epifanía del Señor"),
        (date(año, 5, 1), "Día del Trabajo"),
        (date(año, 8, 15), "Asunción de la Virgen"),
        (date(año, 10, 12), "Fiesta Nacional de España"),
        (date(año, 12, 8), "Inmaculada Concepción"),
        (date(año, 12, 25), "Navidad"),
    ]

    for fecha, nombre in fijos:
        festivos[fecha] = (nombre, "nacional")

    # --------------------------------------------------------
    # 2. FESTIVOS NACIONALES TRASLADABLES
    # --------------------------------------------------------
    # Cuando caen en domingo se trasladan al lunes siguiente.

    trasladables = [
        (date(año, 11, 1), "Todos los Santos"),
        (date(año, 12, 6), "Día de la Constitución"),
    ]

    for fecha, nombre in trasladables:

        festivos[trasladar_si_domingo(fecha)] = (
            nombre,
            "nacional",
        )

    # --------------------------------------------------------
    # 3. FESTIVOS MÓVILES (SEMANA SANTA)
    # --------------------------------------------------------

    festivos[jueves_santo(año)] = ("Jueves Santo", "nacional")
    festivos[viernes_santo(año)] = ("Viernes Santo", "nacional")

    # --------------------------------------------------------
    # 4. FESTIVO DE LA COMUNIDAD DE MADRID
    # --------------------------------------------------------

    festivos[date(año, 5, 2)] = (
        "Día de la Comunidad de Madrid",
        "autonomico",
    )

    # --------------------------------------------------------
    # 5. FIESTAS LOCALES DE ARGANDA DEL REY
    # --------------------------------------------------------
    # Fiestas patronales en honor a la Virgen de la Soledad,
    # celebradas en septiembre.

    festivos[date(año, 9, 8)] = (
        "Virgen de la Soledad (fiesta local)",
        "local",
    )

    festivos[date(año, 9, 9)] = (
        "Fiesta local de Arganda del Rey",
        "local",
    )

    return festivos


def construir_festivos(año_inicio, año_fin):
    """
    Construye el diccionario de festivos de todo el periodo.
    """

    festivos = {}

    for año in range(año_inicio, año_fin + 1):

        festivos.update(
            festivos_del_año(año)
        )

    return festivos


# ============================================================
# PERIODOS VACACIONALES
# ============================================================

def periodos_vacacionales_del_año(año):
    """
    Devuelve los periodos vacacionales del restaurante.

    Los periodos se definen por reglas y no por fechas
    escritas a mano:

        - Navidad:      del 1 al 7 de enero.
        - Semana Santa: del Domingo de Ramos al Domingo
                        siguiente a la Pascua.
        - Verano:       todo el mes de agosto.
    """

    pascua = domingo_de_pascua(año)

    # Domingo de Ramos: una semana antes de la Pascua.
    domingo_de_ramos = pascua - timedelta(days=7)

    # El periodo se cierra el domingo siguiente a la Pascua.
    fin_semana_santa = pascua + timedelta(days=7)

    return [
        {
            "inicio": date(año, 1, 1),
            "fin": date(año, 1, 7),
            "nombre": "Vacaciones de Navidad",
        },
        {
            "inicio": domingo_de_ramos,
            "fin": fin_semana_santa,
            "nombre": "Vacaciones de Semana Santa",
        },
        {
            "inicio": date(año, 8, 1),
            "fin": date(año, 8, 31),
            "nombre": "Vacaciones de verano",
        },
    ]


def construir_periodos_vacacionales(año_inicio, año_fin):
    """
    Construye la lista de periodos vacacionales del periodo
    completo.
    """

    periodos = []

    for año in range(año_inicio, año_fin + 1):

        periodos.extend(
            periodos_vacacionales_del_año(año)
        )

    return periodos


def comprobar_vacaciones(fecha, periodos):
    """
    Comprueba si una fecha pertenece a algún periodo
    vacacional.

    Devuelve:
        (True, nombre_periodo)
    o:
        (False, None)
    """

    for periodo in periodos:

        if periodo["inicio"] <= fecha <= periodo["fin"]:

            return True, periodo["nombre"]

    return False, None


# ============================================================
# GENERAR CALENDARIO
# ============================================================

def generar_calendario(
    año_inicio=AÑO_INICIO,
    año_fin=AÑO_FIN,
):
    """
    Genera el calendario diario completo del periodo indicado.
    """

    # Se construye un año de margen a cada lado para poder
    # mirar el día anterior y el siguiente en los bordes del
    # periodo (vísperas y puentes de fin de año).

    festivos = construir_festivos(
        año_inicio - 1,
        año_fin + 1,
    )

    periodos = construir_periodos_vacacionales(
        año_inicio,
        año_fin,
    )

    filas = []

    fecha_actual = date(año_inicio, 1, 1)
    fecha_final = date(año_fin, 12, 31)

    while fecha_actual <= fecha_final:

        # ----------------------------------------------------
        # DÍA DE LA SEMANA
        # ----------------------------------------------------

        dia_semana_num = fecha_actual.weekday()

        dia_semana = NOMBRES_DIAS[dia_semana_num]

        fin_de_semana = dia_semana_num >= 5

        # ----------------------------------------------------
        # FESTIVO
        # ----------------------------------------------------

        festivo = festivos.get(fecha_actual)

        es_festivo = int(festivo is not None)

        nombre_festivo = festivo[0] if festivo else None
        ambito_festivo = festivo[1] if festivo else None

        # ----------------------------------------------------
        # VÍSPERAS Y PUENTES
        # ----------------------------------------------------
        # Hechos del almanaque, no hipótesis: se leen mirando
        # el día anterior y el siguiente.
        #
        # Puente: día laborable que queda encajonado entre un
        # festivo y el fin de semana. En la práctica española
        # son dos casos, el lunes anterior a un martes festivo
        # y el viernes posterior a un jueves festivo.

        es_vispera_festivo = int(
            (fecha_actual + timedelta(days=1)) in festivos
        )

        es_puente = 0

        if es_festivo == 0 and not fin_de_semana:

            martes_festivo = (
                dia_semana_num == 0
                and (fecha_actual + timedelta(days=1)) in festivos
            )

            jueves_festivo = (
                dia_semana_num == 4
                and (fecha_actual - timedelta(days=1)) in festivos
            )

            if martes_festivo or jueves_festivo:
                es_puente = 1

        # ----------------------------------------------------
        # VACACIONES
        # ----------------------------------------------------

        (
            es_vacaciones,
            periodo_vacacional,
        ) = comprobar_vacaciones(
            fecha_actual,
            periodos,
        )

        # ----------------------------------------------------
        # CIERRE POR DESCANSO SEMANAL
        # ----------------------------------------------------
        # Lunes y martes son los días habituales de descanso
        # del personal.
        #
        # Hay dos excepciones, por criterio de negocio:
        #
        #   - Durante las vacaciones no se cierra, porque la
        #     actividad cambia por completo.
        #
        #   - Tampoco se cierra en festivo. Ningún restaurante
        #     de Arganda echa el cierre el día de la Virgen de
        #     la Soledad porque caiga en martes.

        cerrado_descanso = 0

        if dia_semana_num in DIAS_DESCANSO:

            if not es_vacaciones and es_festivo == 0:
                cerrado_descanso = 1

        # ----------------------------------------------------
        # CIERRE POR VACACIONES DE VERANO
        # ----------------------------------------------------

        cerrado_vacaciones = 0

        if (
            fecha_actual.month == MES_CIERRE_VACACIONES
            and es_vacaciones
        ):
            cerrado_vacaciones = 1

        # ----------------------------------------------------
        # CIERRE DEL RESTAURANTE
        # ----------------------------------------------------

        restaurante_cerrado = 0
        motivo_cierre = None

        if cerrado_vacaciones == 1:

            restaurante_cerrado = 1
            motivo_cierre = "Vacaciones de verano"

        elif cerrado_descanso == 1:

            restaurante_cerrado = 1
            motivo_cierre = "Descanso del personal"

        elif es_festivo == 1 and nombre_festivo == "Navidad":

            # Los festivos se trabajan salvo el día de Navidad.
            restaurante_cerrado = 1
            motivo_cierre = "Navidad"

        # ----------------------------------------------------
        # GUARDAR FILA
        # ----------------------------------------------------

        filas.append({

            "fecha": fecha_actual,
            "año": fecha_actual.year,
            "mes": fecha_actual.month,
            "dia_semana": dia_semana,
            "dia_semana_num": dia_semana_num,
            "fin_de_semana": int(fin_de_semana),

            "es_festivo": es_festivo,
            "nombre_festivo": nombre_festivo,
            "ambito_festivo": ambito_festivo,

            "es_vispera_festivo": es_vispera_festivo,
            "es_puente": es_puente,

            "es_vacaciones": int(es_vacaciones),
            "periodo_vacacional": periodo_vacacional,

            "cerrado_descanso": cerrado_descanso,
            "cerrado_vacaciones": cerrado_vacaciones,

            "restaurante_cerrado": restaurante_cerrado,
            "motivo_cierre": motivo_cierre,
        })

        fecha_actual += timedelta(days=1)

    return pd.DataFrame(filas)


# ============================================================
# GUARDAR CALENDARIO
# ============================================================

def guardar_calendario(calendario):
    """
    Guarda el calendario en data/processed/calendario.
    """

    PROCESSED_CALENDARIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calendario.to_csv(
        CALENDARIO,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Calendario guardado correctamente.")
    print(f"Archivo: {CALENDARIO}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERACIÓN DEL CALENDARIO")
    print("=" * 60)

    print()
    print("Localidad: Arganda del Rey (Comunidad de Madrid)")
    print(f"Periodo:   {AÑO_INICIO} - {AÑO_FIN}")

    calendario = generar_calendario()

    guardar_calendario(calendario)

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print()
    print("RESUMEN")
    print("-" * 60)

    print(
        f"Total de días: {len(calendario)}"
    )

    print(
        f"Días festivos: "
        f"{calendario['es_festivo'].sum()}"
    )

    print(
        f"Días de vacaciones: "
        f"{calendario['es_vacaciones'].sum()}"
    )

    print(
        f"Días cerrado por descanso: "
        f"{calendario['cerrado_descanso'].sum()}"
    )

    print(
        f"Días cerrado por vacaciones: "
        f"{calendario['cerrado_vacaciones'].sum()}"
    )

    print(
        f"Días cerrado en total: "
        f"{calendario['restaurante_cerrado'].sum()}"
    )

    print(
        f"Días de actividad: "
        f"{(calendario['restaurante_cerrado'] == 0).sum()}"
    )

    print(
        f"Vísperas de festivo: "
        f"{calendario['es_vispera_festivo'].sum()}"
    )

    print(
        f"Puentes: "
        f"{calendario['es_puente'].sum()}"
    )

    # --------------------------------------------------------
    # FESTIVOS POR AÑO
    # --------------------------------------------------------

    print()
    print("FESTIVOS POR AÑO")
    print("-" * 60)

    festivos_por_año = (
        calendario[calendario["es_festivo"] == 1]
        .groupby("año")
        .size()
    )

    print(festivos_por_año.to_string())

    print()
    print("FESTIVOS POR ÁMBITO")
    print("-" * 60)

    print(
        calendario[calendario["es_festivo"] == 1]
        ["ambito_festivo"]
        .value_counts()
        .to_string()
    )

    print()
    print("Generación finalizada correctamente.")


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
