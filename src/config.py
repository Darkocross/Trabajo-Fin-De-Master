# src/config.py

# ============================================================
# CONFIGURACIÓN CENTRAL DEL PROYECTO
# ============================================================
#
# Este módulo concentra:
#
#   - Las rutas de todos los directorios y ficheros.
#   - Los parámetros de negocio del restaurante.
#   - Los parámetros de la separación temporal.
#   - La configuración de acceso a AEMET OpenData.
#
# Ningún otro módulo debe construir rutas a mano.
#
# IMPORTANTE:
# La API key de AEMET NUNCA se escribe en el código. Se lee
# del fichero `.env`, que está excluido del control de
# versiones. Usa `.env.example` como plantilla.
#
# ============================================================

from datetime import date
import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# CARGAR VARIABLES DE ENTORNO
# ============================================================

load_dotenv()


# ============================================================
# IDENTIDAD DEL PROYECTO
# ============================================================

# Establecimiento ficticio que sirve de caso de estudio.
# La actividad diaria está generada (ver
# src/transformation/generar_hosteleria.py); la meteorología y
# el calendario son reales.
NOMBRE_RESTAURANTE = "Restaurante de estudio"

LOCALIDAD = "Arganda del Rey"

PROVINCIA = "Madrid"

COMUNIDAD_AUTONOMA = "Comunidad de Madrid"

# ============================================================
# DIRECTORIO RAÍZ DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DIRECTORIOS
# ============================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
MODELOS_DIR = DATA_DIR / "modelos"

DOCS_DIR = ROOT_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

RAW_HOSTELERIA_DIR = RAW_DIR / "hosteleria"

RAW_METEOROLOGIA_DIR = RAW_DIR / "meteorologia"

PROCESSED_CALENDARIO_DIR = PROCESSED_DIR / "calendario"

PROCESSED_MODELADO_DIR = PROCESSED_DIR / "modelado"

PROCESSED_RESULTADOS_DIR = PROCESSED_DIR / "resultados"

PROCESSED_GRAFICOS_DIR = PROCESSED_DIR / "graficos"


# ============================================================
# DATASETS
# ============================================================

# --- Capa raw ---

METEOROLOGIA_HISTORICA = (
    RAW_METEOROLOGIA_DIR / "meteorologia.csv"
)

PREDICCION_AEMET = (
    RAW_METEOROLOGIA_DIR / "prediccion.json"
)

ACTIVIDAD_HOSTELERIA = (
    RAW_HOSTELERIA_DIR / "actividad_hosteleria.csv"
)

# --- Capa processed ---

CALENDARIO = (
    PROCESSED_CALENDARIO_DIR / "calendario.csv"
)

TRAIN = PROCESSED_MODELADO_DIR / "train.csv"
VALIDATION = PROCESSED_MODELADO_DIR / "validation.csv"
TEST = PROCESSED_MODELADO_DIR / "test.csv"

# --- Capa gold ---

GOLD_DATASET = (
    GOLD_DIR / "demanda_restaurante.csv"
)


# ============================================================
# RESULTADOS
# ============================================================

COMPARACION_MODELOS = (
    PROCESSED_RESULTADOS_DIR / "comparacion_modelos.csv"
)

PREDICCIONES_TEST = (
    PROCESSED_RESULTADOS_DIR / "predicciones_test.csv"
)

ANALISIS_ERRORES = (
    PROCESSED_RESULTADOS_DIR / "analisis_errores.csv"
)

PREDICCION_MANANA = (
    PROCESSED_RESULTADOS_DIR / "prediccion_manana.csv"
)


def ruta_importancia(nombre_modelo):
    """
    Devuelve la ruta del CSV de importancia de variables de
    un modelo.
    """

    return (
        PROCESSED_RESULTADOS_DIR
        / f"importancia_{nombre_modelo}.csv"
    )


def ruta_grafico(nombre):
    """
    Devuelve la ruta de un gráfico generado por el proyecto.
    """

    return PROCESSED_GRAFICOS_DIR / f"{nombre}.png"


# ============================================================
# MODELOS ENTRENADOS
# ============================================================

def ruta_modelo(nombre_modelo):
    """
    Devuelve la ruta del fichero .pkl de un modelo.
    """

    return MODELOS_DIR / f"modelo_{nombre_modelo}.pkl"


# Modelo que utiliza la aplicación y la predicción diaria.
#
# Se elige por el error medio absoluto en validación y se
# confirma con la validación cruzada temporal, que comprueba
# que la ventaja se mantiene en periodos distintos y no es
# fruto de la partición concreta.
#
# La decisión está documentada en:
#   docs/entregas/04_analisis_modelado.md
MODELO_PRODUCCION = "regresion_lineal"

METADATOS_MODELOS = (
    MODELOS_DIR / "metadatos_modelos.json"
)


# ============================================================
# PARÁMETROS DE NEGOCIO
# ============================================================

# Periodo cubierto por el proyecto.
AÑO_INICIO = 2022


# Origen del índice temporal.
#
# Tiene que ser una fecha FIJA, no el primer día que haya en
# los datos. Si dependiera de los datos, el índice cambiaría de
# origen cada vez que el dataset cambia de rango, y el modelo
# entrenado dejaría de entender las fechas que le llegan en
# producción.

FECHA_ORIGEN = date(AÑO_INICIO, 1, 1)
AÑO_FIN = 2026

# Días de descanso semanal (0 = lunes ... 6 = domingo).
DIAS_DESCANSO = (0, 1)

# Mes de cierre por vacaciones de verano.
MES_CIERRE_VACACIONES = 8

# Ratio de clientes por empleado usado para recomendar
# plantilla. Procede de los umbrales con los que se etiqueta
# la variable `nota_faena` en los datos de actividad.
RATIO_CLIENTES_EMPLEADO_OBJETIVO = 18

RATIO_FALTA_PERSONAL = 24

RATIO_PERSONAL_OCUPADO = 15

# Límites de plantilla del establecimiento.
EMPLEADOS_MINIMO = 3
EMPLEADOS_MAXIMO = 10


# ============================================================
# PARÁMETROS ECONÓMICOS
# ============================================================
#
# Se usan para traducir el error de predicción en euros, en
# `src/analysis/evaluacion_decision.py`.
#
# NO son datos del establecimiento: son referencias públicas
# del sector, y por eso el análisis va acompañado siempre de
# un estudio de sensibilidad. Las conclusiones no deben
# depender del valor exacto de ninguno de estos números.

# Coste para la empresa de una jornada de camarero.
#
# Convenio de hostelería de la Comunidad de Madrid: unos
# 20.300 € brutos anuales para 1.800 horas. Con la cotización
# empresarial (~32 %) salen unos 14,9 €/hora, es decir, unos
# 119 € por jornada de 8 horas.
COSTE_JORNADA_EMPLEADO = 119.0

# Ticket medio en restauración en España (~21 €) menos el
# coste de materia prima de la hostelería tradicional
# (28-32 %). Queda un margen de contribución de unos 14,5 €
# por cliente.
MARGEN_POR_CLIENTE = 14.5

# Fracción de la demanda que excede la capacidad del personal
# y que se pierde de verdad: gente que se va sin sentarse, que
# no repite, o que consume menos por un servicio peor.
#
# Es el parámetro más incierto de todos, y por eso el análisis
# lo recorre de 0,25 a 1,00.
FRACCION_DEMANDA_PERDIDA = 0.5


# ============================================================
# SEPARACIÓN TEMPORAL
# ============================================================
#
# La separación es temporal y no aleatoria: entrenar con datos
# posteriores a los de evaluación produciría una estimación
# optimista e irreal del error.

AÑOS_TRAIN = (2022, 2023, 2024)

AÑO_VALIDACION = 2025

AÑO_TEST = 2026


# ============================================================
# CONFIGURACIÓN AEMET
# ============================================================

# Estación meteorológica del histórico observado.
# 3182Y -> Arganda del Rey (Madrid)
AEMET_ESTACION_HISTORICO = "3182Y"

# Municipio para la predicción a 7 días.
# 28014 -> Arganda del Rey (código INE)
AEMET_MUNICIPIO_PREDICCION = "28014"

# Alias mantenido por compatibilidad con versiones anteriores
# del proyecto.
AEMET_ESTACION_PREDICCION = AEMET_MUNICIPIO_PREDICCION

AEMET_BASE_URL = "https://opendata.aemet.es"

AEMET_TIMEOUT = 60


# ============================================================
# API KEY AEMET
# ============================================================
#
# Se solicita gratuitamente en:
#   https://opendata.aemet.es/centrodedescargas/altaUsuario
#
# y se guarda en el fichero `.env` de la raíz del proyecto:
#
#   AEMET_API_KEY=tu_api_key

AEMET_API_KEY = os.getenv("AEMET_API_KEY", "")


# ============================================================
# SEMILLA ALEATORIA
# ============================================================
#
# Se fija para que todo el proyecto sea reproducible.

SEMILLA = 42

