# src/transformation/crear_gold.py

# ============================================================
# CONSTRUCCIÓN DE LA CAPA GOLD
# ============================================================
#
# Une las tres fuentes del proyecto en un único dataset
# analítico:
#
#   raw/hosteleria/actividad_hosteleria.csv   (actividad)
#   processed/calendario/calendario.csv       (calendario)
#   raw/meteorologia/meteorologia.csv         (meteorología)
#
# Resultado:
#
#   gold/demanda_restaurante.csv
#
# Granularidad: una fila por día de actividad del restaurante.
#
# Este módulo NO calcula variables de calendario: las toma tal
# cual del calendario, que es su única fuente de verdad. Su
# responsabilidad es unir, filtrar y validar.
#
# ============================================================

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    FECHA_ORIGEN,
    ACTIVIDAD_HOSTELERIA,
    CALENDARIO,
    GOLD_DATASET,
    METEOROLOGIA_HISTORICA,
)


# ============================================================
# COLUMNAS DE ENTRADA
# ============================================================

COLUMNAS_HOSTELERIA = [
    "fecha",
    "n_empleados",
    "n_clientes",
    "nota_faena",
]


COLUMNAS_CALENDARIO = [
    "fecha",
    "año",
    "mes",
    "dia_semana",
    "dia_semana_num",
    "fin_de_semana",
    "es_festivo",
    "nombre_festivo",
    "ambito_festivo",
    "es_vispera_festivo",
    "es_puente",
    "es_vacaciones",
    "periodo_vacacional",
    "restaurante_cerrado",
]


COLUMNAS_METEOROLOGIA = [
    "fecha",
    "tmed",
    "tmax",
    "tmin",
    "prec",
]


# ============================================================
# COLUMNAS DE SALIDA
# ============================================================
#
# Se fija el orden para que la capa gold sea estable y fácil
# de leer, tanto por una persona como por los tests.

COLUMNAS_GOLD = [
    # Identificación temporal
    "fecha",
    "año",
    "mes",
    "dia_semana",
    "nombre_dia",
    "fin_de_semana",
    "indice_temporal",

    # Calendario
    "es_festivo",
    "nombre_festivo",
    "ambito_festivo",
    "es_vispera_festivo",
    "es_puente",
    "es_vacaciones",
    "periodo_vacacional",
    "tipo_vacaciones",

    # Meteorología
    "tmed",
    "tmax",
    "tmin",
    "prec",
    "prec_inapreciable",
    "temp_interpolada",
    "prec_rellenada",
    "meteo_imputada",

    # Actividad
    "n_empleados",
    "n_clientes",
    "nota_faena",
]


# ============================================================
# CARGA DE FICHEROS
# ============================================================

def cargar_csv(ruta, nombre, columnas):
    """
    Carga un CSV del proyecto y comprueba que contiene las
    columnas esperadas.
    """

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existe el fichero de {nombre}:\n{ruta}"
        )

    df = pd.read_csv(
        ruta,
        parse_dates=["fecha"],
    )

    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            f"Faltan columnas en {nombre}: {faltantes}"
        )

    print(f"{nombre:<15} {len(df):>5} registros")

    return df[columnas].copy()


# ============================================================
# TRATAMIENTO DE HUECOS METEOROLÓGICOS
# ============================================================

def interpretar_medida(serie):
    """
    Convierte una columna de medida de AEMET en número,
    interpretando explícitamente sus códigos.

    La capa raw guarda lo que AEMET envía, incluidos los
    códigos. Aquí es donde se decide qué significan, que es una
    decisión de análisis y por tanto debe estar documentada:

        "Ip"    -> 0.0   Precipitación inapreciable. SÍ llovió,
                         pero por debajo de 0,1 mm. Es una
                         MEDICIÓN REAL, no un hueco: el valor
                         numérico más fiel es 0,0 mm.

        "Acum"  -> NaN   El dato está acumulado en otra fecha.
                         Para este día no hay medida propia.

        ""      -> NaN   No hay medición.

    Devuelve la serie numérica y una máscara con los días que
    eran "Ip", para poder marcarlos sin confundirlos con una
    imputación.
    """

    texto = (
        serie
        .astype("string")
        .str.strip()
    )

    # Ojo: la comparación deja <NA> donde el valor es nulo, y
    # `mask` trata <NA> como True, lo que convertiría los
    # huecos en ceros. Se rellena ANTES de usarla.

    inapreciable = (
        (texto.str.lower() == "ip")
        .fillna(False)
        .astype(bool)
    )

    numerico = pd.to_numeric(
        texto.where(~inapreciable),
        errors="coerce",
    )

    numerico = numerico.mask(inapreciable, 0.0)

    # A float64 de toda la vida, no al Float64 anulable de
    # pandas. El anulable se propaga a todo lo que toque esta
    # columna, y numpy no sabe manejarlo: np.corrcoef sobre un
    # DataFrame con dtypes anulables falla con un error que no
    # dice nada ("'float' object has no attribute 'shape'").
    # Aquí no aporta nada, porque los huecos se completan
    # inmediatamente después.

    numerico = numerico.astype("float64")

    return numerico, inapreciable


def completar_meteorologia(meteorologia):
    """
    Interpreta los códigos de AEMET y completa los huecos que
    quedan.

    La estación 3182Y tiene días sin dato. Son datos reales de
    un organismo público, y descartar esos días significaría
    perder también la actividad registrada del restaurante. Se
    completan, pero dejando constancia de cuáles y por qué.

    Tres situaciones DISTINTAS, que una versión anterior del
    proyecto confundía en una sola bandera:

    1. PRECIPITACIÓN INAPRECIABLE ("Ip").
       No es un hueco: es una medición. Llovió menos de
       0,1 mm. Se traduce a 0,0 mm y se marca en
       `prec_inapreciable`. NO cuenta como imputación.

    2. TEMPERATURA AUSENTE.
       Se interpola en el tiempo: la temperatura está muy
       autocorrelacionada de un día para otro, así que para un
       hueco de uno o dos días el valor interpolado es una
       estimación razonable. Se marca en `temp_interpolada`.

    3. PRECIPITACIÓN AUSENTE.
       Se rellena con 0,0 mm y se marca en `prec_rellenada`.

    Sobre el punto 3 conviene ser honesto, porque es la
    decisión más discutible del proyecto. Rellenar con 0 la
    lluvia ausente equivale a afirmar que no llovió, y eso no
    se sabe. Es defendible para huecos de un día suelto, y es
    una afirmación fuerte para un bloque largo.

    El histórico tiene un bloque así: del 19/10/2022 al
    20/11/2022, 33 días seguidos sin precipitación y con todas
    las temperaturas correctas. No es lluvia inapreciable, es
    el pluviómetro sin dar dato durante un mes. De ahí que la
    bandera exista y que la memoria lo declare como limitación.

    `meteo_imputada` vale 1 cuando algún número de ese día lo
    hemos puesto nosotros, es decir, temperatura interpolada o
    precipitación rellenada. Los días "Ip" no entran, porque
    ahí el dato es de AEMET.
    """

    meteorologia = meteorologia.copy()

    meteorologia = (
        meteorologia
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    columnas_temperatura = ["tmed", "tmax", "tmin"]

    # --------------------------------------------------------
    # 1. INTERPRETAR LOS CÓDIGOS DE AEMET
    # --------------------------------------------------------

    inapreciable = pd.Series(
        False,
        index=meteorologia.index,
    )

    for columna in columnas_temperatura + ["prec"]:

        numerico, marca = interpretar_medida(
            meteorologia[columna]
        )

        meteorologia[columna] = numerico

        if columna == "prec":
            inapreciable = marca

    meteorologia["prec_inapreciable"] = (
        inapreciable.astype(int)
    )

    # --------------------------------------------------------
    # 2. MARCAR LO QUE VAMOS A INVENTAR
    # --------------------------------------------------------

    meteorologia["temp_interpolada"] = (
        meteorologia[columnas_temperatura]
        .isna()
        .any(axis=1)
        .astype(int)
    )

    meteorologia["prec_rellenada"] = (
        meteorologia["prec"]
        .isna()
        .astype(int)
    )

    meteorologia["meteo_imputada"] = (
        (
            (meteorologia["temp_interpolada"] == 1)
            | (meteorologia["prec_rellenada"] == 1)
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # 3. COMPLETAR
    # --------------------------------------------------------

    meteorologia = meteorologia.set_index("fecha")

    meteorologia[columnas_temperatura] = (
        meteorologia[columnas_temperatura]
        .interpolate(method="time", limit_direction="both")
    )

    meteorologia = meteorologia.reset_index()

    meteorologia["prec"] = meteorologia["prec"].fillna(0.0)

    # --------------------------------------------------------
    # 4. INFORME
    # --------------------------------------------------------

    dias_ip = int(meteorologia["prec_inapreciable"].sum())
    dias_temp = int(meteorologia["temp_interpolada"].sum())
    dias_prec = int(meteorologia["prec_rellenada"].sum())

    if dias_ip:
        print(
            f"Precipitación inapreciable (Ip, dato de AEMET): "
            f"{dias_ip} días -> 0,0 mm"
        )

    if dias_temp or dias_prec:

        print(
            f"Huecos completados por nosotros: "
            f"{dias_temp} de temperatura (interpolación), "
            f"{dias_prec} de precipitación (0,0 mm)"
        )

        avisar_bloques_largos(meteorologia)

    return meteorologia


def avisar_bloques_largos(meteorologia, minimo=5):
    """
    Avisa de rachas largas de precipitación rellenada.

    Un día suelto sin dato se rellena sin mayor problema. Una
    racha de semanas significa que el sensor estaba caído, y
    poner 0 mm en todos esos días es una afirmación fuerte que
    conviene ver en pantalla y no descubrir en la defensa.
    """

    relleno = meteorologia["prec_rellenada"] == 1

    if not relleno.any():
        return

    grupo = (relleno != relleno.shift()).cumsum()

    rachas = (
        meteorologia[relleno]
        .groupby(grupo[relleno])["fecha"]
        .agg(["min", "max", "size"])
    )

    largas = rachas[rachas["size"] >= minimo]

    if largas.empty:
        return

    print()
    print(
        "AVISO: rachas largas sin precipitación medida. "
        "Se han rellenado con 0,0 mm,"
    )
    print(
        "pero eso equivale a afirmar que no llovió, "
        "y eso no se sabe."
    )

    for _, fila in largas.iterrows():

        print(
            f"  {fila['min'].date()} -> {fila['max'].date()}"
            f"   ({int(fila['size'])} días)"
        )


# ============================================================
# TIPO DE VACACIONES
# ============================================================

def obtener_tipo_vacaciones(periodo):
    """
    Simplifica el nombre del periodo vacacional en una
    categoría manejable por el modelo.
    """

    if pd.isna(periodo):
        return "ninguna"

    periodo = str(periodo).strip().lower()

    if periodo in ("", "nan", "none"):
        return "ninguna"

    if "navidad" in periodo:
        return "navidad"

    if "semana santa" in periodo:
        return "semana_santa"

    if "verano" in periodo:
        return "verano"

    return "otros"


# ============================================================
# CONSTRUCCIÓN DE LA CAPA GOLD
# ============================================================

def crear_gold(hosteleria, calendario, meteorologia):
    """
    Une las tres fuentes y construye la capa gold.
    """

    # --------------------------------------------------------
    # 1. UNIR ACTIVIDAD Y CALENDARIO
    # --------------------------------------------------------
    #
    # El calendario es la referencia temporal: define todos los
    # días del periodo, incluyendo los de cierre.

    print()
    print("Uniendo actividad y calendario...")

    gold = calendario.merge(
        hosteleria,
        on="fecha",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 2. FILTRAR DÍAS DE CIERRE
    # --------------------------------------------------------
    #
    # Un día cerrado no es un día de demanda cero: es un día
    # sin observación. Incluirlo hundiría artificialmente la
    # media y enseñaría al modelo un patrón que no existe.

    dias_cerrados = int(
        (gold["restaurante_cerrado"] == 1).sum()
    )

    gold = gold[
        gold["restaurante_cerrado"] == 0
    ].copy()

    gold = gold.drop(columns=["restaurante_cerrado"])

    print(
        f"Días de cierre excluidos: {dias_cerrados}"
    )

    # --------------------------------------------------------
    # 3. UNIR METEOROLOGÍA
    # --------------------------------------------------------

    print("Uniendo meteorología...")

    gold = gold.merge(
        meteorologia,
        on="fecha",
        how="left",
        validate="one_to_one",
    )

    columnas_meteo = ["tmed", "tmax", "tmin", "prec"]

    sin_meteo = int(
        gold[columnas_meteo].isna().all(axis=1).sum()
    )

    if sin_meteo > 0:

        print(
            f"Días sin meteorología: {sin_meteo} "
            "(se excluyen)"
        )

        gold = gold[
            ~gold[columnas_meteo].isna().all(axis=1)
        ].copy()

    # --------------------------------------------------------
    # 4. DÍAS SIN ACTIVIDAD REGISTRADA
    # --------------------------------------------------------

    sin_actividad = int(gold["n_clientes"].isna().sum())

    if sin_actividad > 0:

        print(
            f"Días abiertos sin actividad registrada: "
            f"{sin_actividad} (se excluyen)"
        )

        gold = gold[
            gold["n_clientes"].notna()
        ].copy()

    # --------------------------------------------------------
    # 5. VARIABLES DERIVADAS
    # --------------------------------------------------------

    print("Calculando variables derivadas...")

    # Nombre del día (para informes y gráficos).
    gold = gold.rename(
        columns={
            "dia_semana": "nombre_dia",
            "dia_semana_num": "dia_semana",
        }
    )

    # Categoría de vacaciones.
    gold["tipo_vacaciones"] = (
        gold["periodo_vacacional"]
        .apply(obtener_tipo_vacaciones)
    )

    # Índice temporal: días transcurridos desde una fecha FIJA
    # del proyecto, no desde el primer día del dataset.
    #
    # La diferencia importa. Si el origen dependiera de los
    # datos, bastaría con que cambiara el rango del histórico
    # para que el mismo día tuviera un índice distinto, y el
    # modelo ya entrenado interpretaría mal las fechas nuevas.
    #
    # Es la variable que le dice al modelo que ha pasado el
    # tiempo, y con ella se corrige el sesgo por crecimiento
    # del negocio. Ver src/models/features.py.
    gold["indice_temporal"] = (
        gold["fecha"] - pd.Timestamp(FECHA_ORIGEN)
    ).dt.days

    # --------------------------------------------------------
    # 6. TIPOS
    # --------------------------------------------------------

    columnas_enteras = [
        "año",
        "mes",
        "dia_semana",
        "fin_de_semana",
        "indice_temporal",
        "es_festivo",
        "es_vispera_festivo",
        "es_puente",
        "es_vacaciones",
        "n_empleados",
        "n_clientes",
        "prec_inapreciable",
        "temp_interpolada",
        "prec_rellenada",
        "meteo_imputada",
    ]

    for columna in columnas_enteras:

        gold[columna] = (
            pd.to_numeric(gold[columna], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    columnas_decimales = ["tmed", "tmax", "tmin", "prec"]

    for columna in columnas_decimales:

        gold[columna] = pd.to_numeric(
            gold[columna],
            errors="coerce",
        )

    gold["nombre_festivo"] = (
        gold["nombre_festivo"].fillna("")
    )

    gold["periodo_vacacional"] = (
        gold["periodo_vacacional"].fillna("ninguna")
    )

    # --------------------------------------------------------
    # 7. ORDENAR Y SELECCIONAR
    # --------------------------------------------------------

    gold = (
        gold[COLUMNAS_GOLD]
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    return gold


# ============================================================
# VALIDACIONES BÁSICAS
# ============================================================

def validar_gold(gold):
    """
    Comprobaciones mínimas antes de guardar. Si alguna falla,
    el pipeline se detiene: es preferible no generar la capa
    gold a generarla mal.
    """

    print()
    print("Validando la capa gold...")

    if gold.empty:

        raise ValueError(
            "La capa gold está vacía."
        )

    duplicados = int(gold["fecha"].duplicated().sum())

    if duplicados > 0:

        raise ValueError(
            f"Existen {duplicados} fechas duplicadas."
        )

    if (gold["n_clientes"] < 0).any():

        raise ValueError(
            "Existen días con un número negativo de clientes."
        )

    columnas_criticas = [
        "fecha",
        "dia_semana",
        "mes",
        "n_clientes",
        "tmed",
        "prec",
    ]

    nulos = gold[columnas_criticas].isna().sum()

    if nulos.sum() > 0:

        raise ValueError(
            "Existen valores nulos en columnas críticas:\n"
            f"{nulos[nulos > 0]}"
        )

    print("Validación superada.")


# ============================================================
# GUARDAR
# ============================================================

def guardar_gold(gold):
    """
    Guarda la capa gold.
    """

    GOLD_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gold.to_csv(
        GOLD_DATASET,
        index=False,
        encoding="utf-8",
    )

    print()
    print("Capa gold guardada correctamente.")
    print(f"Archivo: {GOLD_DATASET}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREACIÓN DE LA CAPA GOLD")
    print("=" * 60)

    print()
    print("Cargando fuentes...")
    print("-" * 60)

    hosteleria = cargar_csv(
        ACTIVIDAD_HOSTELERIA,
        "Hostelería",
        COLUMNAS_HOSTELERIA,
    )

    calendario = cargar_csv(
        CALENDARIO,
        "Calendario",
        COLUMNAS_CALENDARIO,
    )

    meteorologia = cargar_csv(
        METEOROLOGIA_HISTORICA,
        "Meteorología",
        COLUMNAS_METEOROLOGIA,
    )

    print()
    print("Completando huecos meteorológicos...")

    meteorologia = completar_meteorologia(meteorologia)

    gold = crear_gold(
        hosteleria,
        calendario,
        meteorologia,
    )

    validar_gold(gold)

    guardar_gold(gold)

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print()
    print("RESUMEN")
    print("-" * 60)

    print(
        f"Registros: {len(gold)}"
    )

    print(
        f"Periodo:   "
        f"{gold['fecha'].min().date()} -> "
        f"{gold['fecha'].max().date()}"
    )

    print(
        f"Columnas:  {len(gold.columns)}"
    )

    print()
    print("Clientes por día:")
    print(
        f"  Media:  {gold['n_clientes'].mean():.1f}"
    )
    print(
        f"  Mínimo: {gold['n_clientes'].min()}"
    )
    print(
        f"  Máximo: {gold['n_clientes'].max()}"
    )

    print()
    print("Registros por año:")
    print(
        gold.groupby("año").size().to_string()
    )

    print()
    print("=" * 60)
    print("CAPA GOLD CREADA CORRECTAMENTE")
    print("=" * 60)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
