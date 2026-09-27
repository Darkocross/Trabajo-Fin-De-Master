# src/models/features.py

# ============================================================
# DEFINICIÓN DE VARIABLES Y PREPROCESAMIENTO
# ============================================================
#
# Este módulo es la ÚNICA fuente de verdad sobre:
#
#   - Qué variable se predice.
#   - Qué variables se usan como entrada.
#   - Cómo se transforman antes de entrar en el modelo.
#   - Cómo se separan los datos en train / validación / test.
#
# Todos los modelos, el análisis de importancia, la evaluación
# y la aplicación importan de aquí. De este modo es imposible
# que dos partes del proyecto entrenen o evalúen con conjuntos
# de variables distintos.
#
# ------------------------------------------------------------
# CRITERIOS DE SELECCIÓN DE VARIABLES
# ------------------------------------------------------------
#
# 1. Solo información disponible ANTES del servicio.
#
#    `nota_faena` describe cómo ha ido la jornada: es un
#    resultado, no una entrada. Usarla sería fuga de
#    información (leakage).
#
# 2. Sin variables perfectamente colineales.
#
#    Se han eliminado del conjunto de entrada varias variables
#    que son función exacta de otras ya presentes:
#
#      - `fin_de_semana` = dia_semana in {5, 6}
#      - `es_vacaciones` = tipo_vacaciones != "ninguna"
#
#    Mantenerlas no aporta información y desestabiliza los
#    coeficientes de la regresión lineal, que es precisamente
#    el modelo que queremos poder interpretar.
#
# 2 bis. Las INTERACCIONES se construyen AQUÍ, no en la capa
#    de datos, y se construyen de forma sistemática.
#
#    Este módulo es el sitio donde viven las hipótesis de
#    modelado. El calendario describe el almanaque; la capa
#    gold une fuentes; aquí es donde se puede plantear "un
#    sábado de junio no es la suma de sábado y junio", medirlo
#    y quedárselo o tirarlo.
#
#    Una versión anterior del proyecto hacía justo lo
#    contrario: el calendario emitía `es_fin_semana_mayo`,
#    `es_fin_semana_junio` y `es_fin_semana_julio`. Tres meses
#    elegidos a dedo, y elegidos además por el motivo
#    equivocado: eran los tres meses que el generador
#    sintético amplifica. Fuga conceptual.
#
#    Aquí la interacción se plantea para los DOCE meses
#    (`mes_finde`, ver más abajo) y se mide. El resultado de
#    esa medición está en src/analysis/comparar_variables.py.
#
# 3. `n_empleados` NO se utiliza como variable de entrada.
#
#    Se genera a partir del calendario (día de la semana,
#    festivo, Semana Santa) y por tanto no añade información
#    que el modelo no tenga ya. Además, la dirección útil para
#    el negocio es la contraria: primero se estiman los
#    clientes y a partir de ahí se recomienda la plantilla.
#    Usarla como entrada haría el producto circular.
#
# 4. La temperatura entra como NIVEL y AMPLITUD, no como
#    tres columnas.
#
#    AEMET no mide la temperatura media: la calcula como
#    (tmax + tmin) / 2. Se ha comprobado en los datos del
#    proyecto y la identidad se cumple con una desviación de
#    0,04 grados.
#
#    Es decir, `tmed`, `tmax` y `tmin` solo tienen DOS grados
#    de libertad: la tercera es combinación lineal exacta de
#    las otras dos. Meter las tres en una regresión lineal
#    produce un VIF (factor de inflación de la varianza)
#    superior a 45.000, cuando por encima de 10 ya se
#    considera un problema. El resultado eran coeficientes
#    absurdos que se anulaban entre sí:
#
#        temperatura media   -104,2
#        temperatura máxima   +58,1
#        temperatura mínima   +56,8
#
#    Ninguno de esos números significa nada por separado.
#
#    La solución es cambiar de parametrización sin perder
#    información:
#
#        tmed              -> nivel térmico del día
#        amplitud_termica  -> tmax - tmin
#
#    La transformación es invertible (tmax = tmed + amp/2,
#    tmin = tmed - amp/2), así que los árboles conservan toda
#    la información y pueden seguir encontrando los umbrales
#    de calor y frío extremo. Pero la correlación entre ambas
#    baja a 0,47 y el VIF a 1,28, con lo que los coeficientes
#    de la regresión vuelven a ser interpretables.
#
# 5. Transformaciones documentadas de la meteorología.
#
#    - `prec_log`: la precipitación tiene una distribución muy
#      asimétrica (la mayoría de días son 0 mm y unos pocos
#      acumulan mucho). log(1 + prec) es la transformación
#      estándar para este tipo de variable.
#
#    - `lluvia_fin_semana`: hipótesis de negocio recogida ya en
#      la Entrega 2. La lluvia penaliza más un sábado, cuando
#      la clientela es de ocio y puede quedarse en casa, que un
#      jueves, cuando es de trabajo y va igualmente.
#
# ============================================================

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AÑOS_TRAIN,
    AÑO_TEST,
    AÑO_VALIDACION,
)


# ============================================================
# VARIABLE OBJETIVO
# ============================================================

VARIABLE_OBJETIVO = "n_clientes"


# ============================================================
# VARIABLES CATEGÓRICAS
# ============================================================

# ------------------------------------------------------------
# INTERRUPTORES DE VARIABLES OPCIONALES
# ------------------------------------------------------------
#
# Dos bloques de variables están construidos y disponibles,
# pero apagados por defecto. No es dejadez: es el resultado de
# haberlos medido (src/analysis/comparar_variables.py).
#
# USAR_INTERACCION_MES_FINDE
#
#   Añade `mes_finde` (mes × fin de semana, 24 niveles), la
#   versión sistemática de las antiguas `es_fin_semana_*`.
#   Sobre estos datos pasa de 23 a 44 columnas y NO mejora de
#   forma consistente: el MAE de validación se mueve dentro de
#   una desviación típica del ruido de la validación cruzada.
#   Con datos reales de un local, donde el efecto terraza
#   puede ser mucho mayor, merece la pena volver a medirlo.
#
# USAR_VISPERAS_Y_PUENTES
#
#   Añade `es_vispera_festivo` y `es_puente`, que el
#   calendario sí publica porque son hechos del almanaque.
#   Sobre los datos sintéticos son inertes por construcción:
#   el generador no produce ningún efecto de víspera ni de
#   puente (medido: 1,068, indistinguible de ruido). Con datos
#   reales es de las primeras cosas que hay que encender.
#
# Cambiar cualquiera de los dos y volver a lanzar el pipeline
# es todo lo que hace falta.

USAR_INTERACCION_MES_FINDE = False

USAR_VISPERAS_Y_PUENTES = False


# ------------------------------------------------------------
# TENDENCIA TEMPORAL
# ------------------------------------------------------------
#
# `indice_temporal` son los días transcurridos desde el origen
# fijo del proyecto. Es la única variable capaz de decirle al
# modelo que ha pasado el tiempo.
#
# Está APAGADA, y el motivo es el hallazgo más contraintuitivo
# del proyecto. Merece la pena contarlo entero.
#
# EL PROBLEMA. El modelo se queda corto de forma sistemática:
# unos 10 clientes de media en test. El negocio crece en torno
# a un 4 % anual y ninguna variable se lo dice. La memoria lo
# declaraba como "la limitación más relevante y la que primero
# habría que atacar".
#
# LO QUE PARECÍA LA SOLUCIÓN. Añadir la variable arregla el
# sesgo casi por completo y baja el error:
#
#                  MAE test      sesgo medio
#   sin tendencia    17,87          +9,95
#   con tendencia    17,32          +0,82
#
# Con esas dos columnas, la decisión estaba tomada.
#
# LO QUE PASA AL MIRAR LA DECISIÓN. El proyecto entero defiende
# que un modelo se juzga por las decisiones que produce, no por
# su error medio. Aplicando su propio criterio:
#
#                 MAE    sesgo   plantilla OK   falta   exceso
#   sin tendencia 17,87  +9,95      93,4 %        7        4
#   con tendencia 17,32  +0,82      86,2 %        3       20
#
# Mejor error, mejor sesgo y PEOR decisión. Siete puntos peor.
# En euros, de 1.368 € a 2.532 €: casi el doble.
#
# POR QUÉ. La plantilla se decide en números enteros de
# personas, y "acertar" significa caer dentro de una banda de
# entre 15 y 24 clientes por empleado. Un sesgo de 10 clientes
# es aproximadamente medio empleado: casi siempre cabe dentro
# de la banda. Al corregirlo hacia arriba, muchos días cruzan
# el escalón y pasan a llevar una persona de más.
#
# Dicho de otro modo: el sesgo existe, está medido, y resulta
# ser INOCUO para la decisión. Corregirlo hace daño.
#
# Y HAY UN SEGUNDO MOTIVO. Una recta extrapolada supone que el
# crecimiento continúa igual. No continúa: con horizonte largo
# el modelo pasa de quedarse corto a pasarse.
#
#                        sesgo sin    sesgo con
#   entrena 24, evalúa 25   +5,73        -3,96
#   entrena 25, evalúa 26   +9,95        +0,82
#   entrena 24, evalúa 26  +11,23        -3,43
#
# Solo acierta cuando el horizonte es corto. El resto del
# tiempo cambia un error sistemático por otro de signo
# contrario, que además es el caro al revés.
#
# La medición confirmó también lo que la teoría anticipaba:
# la variable solo sirve en el modelo que puede extrapolar.
# Los árboles no pueden, y con ella reducen el sesgo a medias
# sin mejorar el error.
#
# QUÉ SE HACE ENTONCES. Nada en el modelo, y vigilancia fuera
# de él: src/analysis/monitorizacion.py sigue el sesgo y avisa
# cuando deje de ser inocuo, que es cuando empiece a sacar
# decisiones de la banda. La comparación completa, con las
# cuatro opciones probadas y los dos horizontes, está en
# src/analysis/correccion_tendencia.py y se reproduce con
# `python pipeline.py --solo tendencia`.

USAR_TENDENCIA_TEMPORAL = False


VARIABLES_CATEGORICAS = [
    "dia_semana",
    "mes",
    "tipo_vacaciones",
]

if USAR_INTERACCION_MES_FINDE:
    VARIABLES_CATEGORICAS.append("mes_finde")


# ============================================================
# VARIABLES NUMÉRICAS Y BINARIAS
# ============================================================

VARIABLES_NUMERICAS = [
    # Calendario
    "es_festivo",

    # Meteorología
    "tmed",
    "amplitud_termica",
    "prec_log",
    "lluvia_fin_semana",
]

if USAR_VISPERAS_Y_PUENTES:
    VARIABLES_NUMERICAS[1:1] = [
        "es_vispera_festivo",
        "es_puente",
    ]

if USAR_TENDENCIA_TEMPORAL:
    VARIABLES_NUMERICAS.append("indice_temporal")


VARIABLES = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS


# ============================================================
# COLUMNAS QUE DEBEN EXISTIR EN EL DATASET DE ENTRADA
# ============================================================
#
# `prec_log` y `lluvia_fin_semana` no están en la capa gold:
# se derivan aquí con `anadir_variables_derivadas`.

COLUMNAS_REQUERIDAS = [
    "fecha",
    "dia_semana",
    "mes",
    "tipo_vacaciones",
    "fin_de_semana",
    "es_festivo",
    "tmed",
    "tmax",
    "tmin",
    "prec",
]

if USAR_VISPERAS_Y_PUENTES:
    COLUMNAS_REQUERIDAS += [
        "es_vispera_festivo",
        "es_puente",
    ]

if USAR_TENDENCIA_TEMPORAL:
    COLUMNAS_REQUERIDAS.append("indice_temporal")


# ============================================================
# NOMBRES LEGIBLES DE LAS VARIABLES
# ============================================================
#
# Se usan en los informes, los gráficos de importancia y la
# explicación que muestra la aplicación al usuario.

NOMBRES_DIAS = {
    "0": "lunes",
    "1": "martes",
    "2": "miércoles",
    "3": "jueves",
    "4": "viernes",
    "5": "sábado",
    "6": "domingo",
}


NOMBRES_MESES = {
    "1": "enero",
    "2": "febrero",
    "3": "marzo",
    "4": "abril",
    "5": "mayo",
    "6": "junio",
    "7": "julio",
    "8": "agosto",
    "9": "septiembre",
    "10": "octubre",
    "11": "noviembre",
    "12": "diciembre",
}


NOMBRES_VARIABLES = {
    "es_festivo": "es festivo",
    "indice_temporal": "tendencia (días transcurridos)",
    "es_vispera_festivo": "víspera de festivo",
    "es_puente": "puente",
    "tmed": "temperatura media",
    "amplitud_termica": "amplitud térmica",
    "tmax": "temperatura máxima",
    "tmin": "temperatura mínima",
    "prec_log": "precipitación (log)",
    "lluvia_fin_semana": "lluvia en fin de semana",
}


def traducir_variable(nombre):
    """
    Convierte el nombre técnico que genera el preprocesador en
    un nombre legible para un lector no técnico.

    Ejemplos:

        categoricas__dia_semana_5 -> "día de la semana: sábado"
        numericas__tmed           -> "temperatura media"
    """

    nombre = str(nombre)

    # Quitar el prefijo del ColumnTransformer.
    for prefijo in ("categoricas__", "numericas__"):

        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):]

    # --- Día de la semana ---

    if nombre.startswith("dia_semana_"):

        valor = nombre.replace("dia_semana_", "")
        valor = valor.split(".")[0]

        return (
            "día de la semana: "
            f"{NOMBRES_DIAS.get(valor, valor)}"
        )

    # --- Mes x fin de semana ---

    if nombre.startswith("mes_finde_"):

        valor = nombre.replace("mes_finde_", "")

        partes = valor.split("_")

        mes = NOMBRES_MESES.get(partes[0], partes[0])

        tipo = (
            "fin de semana"
            if len(partes) > 1 and partes[1] == "1"
            else "entre semana"
        )

        return f"{mes}, {tipo}"

    # --- Mes ---

    if nombre.startswith("mes_"):

        valor = nombre.replace("mes_", "")
        valor = valor.split(".")[0]

        return f"mes: {NOMBRES_MESES.get(valor, valor)}"

    # --- Tipo de vacaciones ---

    if nombre.startswith("tipo_vacaciones_"):

        valor = nombre.replace("tipo_vacaciones_", "")

        return f"tipo de vacaciones: {valor}"

    # --- Resto ---

    return NOMBRES_VARIABLES.get(nombre, nombre)


# ============================================================
# VARIABLES DERIVADAS
# ============================================================

def anadir_variables_derivadas(df):
    """
    Añade al DataFrame las variables derivadas que necesita el
    modelo y que no están en la capa gold.

    Se aplica igual a train, validación, test y a los datos de
    predicción diaria, de modo que la transformación es
    idéntica en entrenamiento y en producción.
    """

    df = df.copy()

    # --------------------------------------------------------
    # AMPLITUD TÉRMICA
    # --------------------------------------------------------
    #
    # Diferencia entre la máxima y la mínima del día. Junto con
    # `tmed` contiene exactamente la misma información que las
    # tres temperaturas originales, pero sin la colinealidad
    # que las hacía ininterpretables.

    df["amplitud_termica"] = (
        pd.to_numeric(df["tmax"], errors="coerce")
        - pd.to_numeric(df["tmin"], errors="coerce")
    )

    # --------------------------------------------------------
    # PRECIPITACIÓN EN ESCALA LOGARÍTMICA
    # --------------------------------------------------------

    prec = pd.to_numeric(
        df["prec"],
        errors="coerce",
    ).clip(lower=0)

    df["prec_log"] = np.log1p(prec)

    # --------------------------------------------------------
    # INTERACCIÓN LLUVIA x FIN DE SEMANA
    # --------------------------------------------------------

    fin_de_semana = pd.to_numeric(
        df["fin_de_semana"],
        errors="coerce",
    ).fillna(0)

    df["lluvia_fin_semana"] = (
        df["prec_log"] * fin_de_semana
    )

    # --------------------------------------------------------
    # INTERACCIÓN MES x FIN DE SEMANA
    # --------------------------------------------------------
    #
    # La versión sistemática de las antiguas `es_fin_semana_*`:
    # una categoría por cada combinación de mes y tipo de día,
    # veinticuatro en total. Se calcula siempre, aunque el
    # interruptor esté apagado, para que encenderlo no exija
    # tocar nada más.

    if "mes" in df.columns:

        df["mes_finde"] = (
            df["mes"].astype(str)
            + "_"
            + fin_de_semana.astype(int).astype(str)
        )

    return df


# ============================================================
# PREPARACIÓN DEL DATASET
# ============================================================

def preparar_dataset(df, exigir_objetivo=True):
    """
    Deja el DataFrame listo para entrar en el modelo:

        - Convierte tipos.
        - Añade las variables derivadas.
        - Ordena cronológicamente.
        - Elimina filas sin variable objetivo.

    `exigir_objetivo=False` se usa en predicción, donde
    todavía no conocemos el número de clientes.
    """

    df = df.copy()

    # --------------------------------------------------------
    # COMPROBAR COLUMNAS
    # --------------------------------------------------------

    faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "Faltan columnas necesarias para el modelado: "
            f"{faltantes}"
        )

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="coerce",
    )

    if df["fecha"].isna().any():

        raise ValueError(
            f"Existen {int(df['fecha'].isna().sum())} "
            "fechas inválidas."
        )

    # --------------------------------------------------------
    # VARIABLES NUMÉRICAS DE ORIGEN
    # --------------------------------------------------------

    numericas_origen = [
        "es_festivo",
        "fin_de_semana",
        "tmed",
        "tmax",
        "tmin",
        "prec",
    ]

    if USAR_VISPERAS_Y_PUENTES:
        numericas_origen += [
            "es_vispera_festivo",
            "es_puente",
        ]

    if USAR_TENDENCIA_TEMPORAL:
        numericas_origen.append("indice_temporal")

    for columna in numericas_origen:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    # --------------------------------------------------------
    # VARIABLES CATEGÓRICAS
    # --------------------------------------------------------
    #
    # `dia_semana` y `mes` son números, pero se tratan como
    # categorías: el efecto del sábado no es "seis veces" el
    # del lunes, y el de diciembre no es "doce veces" el de
    # enero.

    for columna in ("dia_semana", "mes", "tipo_vacaciones"):

        df[columna] = (
            df[columna]
            .astype("string")
            .fillna("desconocido")
            .astype(str)
        )

    # --------------------------------------------------------
    # VARIABLES DERIVADAS
    # --------------------------------------------------------

    df = anadir_variables_derivadas(df)

    # --------------------------------------------------------
    # VARIABLE OBJETIVO
    # --------------------------------------------------------

    if VARIABLE_OBJETIVO in df.columns:

        df[VARIABLE_OBJETIVO] = pd.to_numeric(
            df[VARIABLE_OBJETIVO],
            errors="coerce",
        )

        if exigir_objetivo:

            df = df.dropna(
                subset=[VARIABLE_OBJETIVO]
            ).copy()

    elif exigir_objetivo:

        raise ValueError(
            "El dataset no contiene la variable objetivo "
            f"'{VARIABLE_OBJETIVO}'."
        )

    # --------------------------------------------------------
    # ORDENAR
    # --------------------------------------------------------

    df = (
        df
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SEPARAR X E Y
# ============================================================

def separar_x_y(df):
    """
    Devuelve la matriz de variables de entrada y el vector
    objetivo, en el orden definido por `VARIABLES`.
    """

    X = df[VARIABLES].copy()

    y = df[VARIABLE_OBJETIVO].copy()

    return X, y


def obtener_x(df):
    """
    Devuelve solo la matriz de entrada. Se usa en predicción,
    donde no existe variable objetivo.
    """

    return df[VARIABLES].copy()


# ============================================================
# SEPARACIÓN TEMPORAL
# ============================================================

def separar_temporal(df):
    """
    Separa el dataset en train, validación y test respetando el
    orden cronológico.

    La separación es la misma en todo el proyecto y está
    definida en `src/config.py`:

        train      -> 2022, 2023 y 2024
        validación -> 2025
        test       -> 2026

    No se usa una separación aleatoria porque introduciría
    información del futuro en el entrenamiento y produciría una
    estimación del error demasiado optimista.
    """

    df = df.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    año = df["fecha"].dt.year

    train = df[año.isin(AÑOS_TRAIN)].copy()

    validacion = df[año == AÑO_VALIDACION].copy()

    test = df[año >= AÑO_TEST].copy()

    return train, validacion, test


def validar_separacion(train, validacion, test):
    """
    Comprueba que los tres conjuntos tienen datos y que no se
    solapan en el tiempo.
    """

    for nombre, conjunto in [
        ("TRAIN", train),
        ("VALIDACIÓN", validacion),
        ("TEST", test),
    ]:

        if conjunto.empty:

            raise ValueError(
                f"El conjunto {nombre} está vacío."
            )

    if train["fecha"].max() >= validacion["fecha"].min():

        raise ValueError(
            "Solapamiento temporal entre TRAIN y VALIDACIÓN."
        )

    if validacion["fecha"].max() >= test["fecha"].min():

        raise ValueError(
            "Solapamiento temporal entre VALIDACIÓN y TEST."
        )


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def crear_preprocesador(escalar=False):
    """
    Construye el preprocesador común a todos los modelos.

    - Categóricas: imputación por moda + One-Hot Encoding.
    - Numéricas:   imputación por mediana.

    `escalar=True` añade estandarización de las variables
    numéricas. Solo la necesita la regresión lineal, para que
    sus coeficientes sean comparables entre sí y se puedan
    interpretar como importancia. Los modelos de árboles son
    insensibles a la escala.

    `drop="first"` elimina una categoría de referencia en cada
    variable categórica para evitar colinealidad perfecta en
    la regresión lineal.
    """

    pasos_categoricas = [
        (
            "imputacion",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            ),
        ),
    ]

    pasos_numericas = [
        (
            "imputacion",
            SimpleImputer(strategy="median"),
        ),
    ]

    if escalar:

        pasos_numericas.append(
            (
                "escalado",
                StandardScaler(),
            )
        )

    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(pasos_categoricas),
                VARIABLES_CATEGORICAS,
            ),
            (
                "numericas",
                Pipeline(pasos_numericas),
                VARIABLES_NUMERICAS,
            ),
        ]
    )


def obtener_nombres_variables(modelo):
    """
    Devuelve los nombres de las columnas que salen del
    preprocesador de un pipeline ya entrenado.
    """

    return (
        modelo
        .named_steps["preprocesamiento"]
        .get_feature_names_out()
    )
