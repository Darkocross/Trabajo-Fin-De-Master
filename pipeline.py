# pipeline.py

# ============================================================
# PIPELINE DEL PROYECTO
# ============================================================
#
# Ejecuta el proyecto completo, de principio a fin, en orden.
#
# Uso:
#
#   python pipeline.py                  Todo salvo la descarga
#   python pipeline.py --todo           Todo, incluida la
#                                       descarga de AEMET
#   python pipeline.py --desde modelos  Desde una etapa
#   python pipeline.py --hasta gold     Hasta una etapa
#   python pipeline.py --solo gold      Una única etapa
#   python pipeline.py --listar         Ver las etapas
#
# Las etapas de descarga (`meteorologia` y `prediccion`) están
# desactivadas por defecto porque requieren conexión y una API
# key de AEMET. Los datos descargados ya están en el
# repositorio, de modo que el pipeline se puede reproducir
# entero sin conexión.
#
# ============================================================

import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# DEFINICIÓN DE LAS ETAPAS
# ============================================================
#
# Cada etapa es una tupla:
#
#     (clave, título, módulo, por_defecto)
#
# `por_defecto = False` significa que la etapa solo se ejecuta
# si se pide expresamente.

ETAPAS = [
    # --- 1. Extracción ---
    (
        "meteorologia",
        "Descarga del histórico meteorológico (AEMET)",
        "src.extraction.descargar_meteorologia",
        False,
    ),
    (
        "prediccion",
        "Descarga de la previsión meteorológica (AEMET)",
        "src.extraction.descargar_prediccion",
        False,
    ),

    # --- 2. Transformación ---
    (
        "calendario",
        "Generación del calendario",
        "src.transformation.generar_calendario",
        True,
    ),
    #(
    #    "hosteleria",
    #    "Descarga de los datos de actividad",
    #    "src.transformation.generar_hosteleria",
    #    True,
    #),
    (
        "duplicados",
        "Limpieza de duplicados",
        "src.transformation.limpiar_duplicados",
        True,
    ),
    (
        "gold",
        "Construcción de la capa gold",
        "src.transformation.crear_gold",
        True,
    ),
    (
        "validar",
        "Validación de la capa gold",
        "src.transformation.validar_gold",
        True,
    ),
    (
        "separar",
        "Separación temporal del dataset",
        "src.transformation.separar_modelo",
        True,
    ),

    # --- 3. Modelado ---
    (
        "baseline",
        "Modelo baseline",
        "src.models.baseline",
        True,
    ),
    (
        "lineal",
        "Regresión lineal",
        "src.models.regresion_lineal",
        True,
    ),
    (
        "bosque",
        "Random Forest",
        "src.models.random_forest",
        True,
    ),
    (
        "boosting",
        "Hist Gradient Boosting",
        "src.models.hist_gradient_boosting",
        True,
    ),
    (
        "cruzada",
        "Validación cruzada temporal",
        "src.models.validacion_cruzada",
        True,
    ),
    (
        "variables",
        "Comparación de conjuntos de variables",
        "src.analysis.comparar_variables",
        True,
    ),
    (
        "produccion",
        "Entrenamiento del modelo de producción",
        "src.models.modelo_produccion",
        True,
    ),

    # --- 4. Análisis ---
    (
        "importancia",
        "Análisis de importancia de las variables",
        "src.models.analisis_importancia",
        True,
    ),
    (
        "errores",
        "Análisis de errores",
        "src.visualization.analisis_errores",
        True,
    ),
    (
        "graficos",
        "Gráficos de evaluación",
        "src.visualization.evaluacion_real_vs_predicho",
        True,
    ),
    (
        "mockup",
        "Mockup del frontal (Entrega 5)",
        "src.visualization.generar_mockup",
        True,
    ),
    (
        "semana",
        "Gráfico de previsión de los próximos días",
        "src.visualization.prevision_semana",
        True,
    ),

    # --- 5. Análisis de valor y validación ---
    (
        "decision",
        "Evaluación como sistema de decisión",
        "src.analysis.evaluacion_decision",
        True,
    ),
    (
        "robustez",
        "Robustez frente al error de previsión",
        "src.analysis.robustez_prevision",
        True,
    ),
    (
        "generador",
        "Validación contra el proceso generador",
        "src.analysis.validacion_generador",
        True,
    ),
    (
        "tendencia",
        "Corrección del sesgo por crecimiento",
        "src.analysis.correccion_tendencia",
        True,
    ),
    (
        "monitorizacion",
        "Monitorización del modelo en producción",
        "src.analysis.monitorizacion",
        True,
    ),

    # --- 6. Predicción ---
    (
        "manana",
        "Predicción de los próximos días",
        "src.models.prediccion_manana",
        True,
    ),
]


CLAVES = [etapa[0] for etapa in ETAPAS]


# ============================================================
# EJECUCIÓN DE UNA ETAPA
# ============================================================

def ejecutar_etapa(numero, total, clave, titulo, modulo):
    """
    Importa el módulo de una etapa y ejecuta su `main()`.
    """

    print()
    print("=" * 70)
    print(f"[{numero}/{total}]  {titulo}")
    print("=" * 70)

    inicio = time.perf_counter()

    modulo_importado = __import__(
        modulo,
        fromlist=["main"],
    )

    modulo_importado.main()

    duracion = time.perf_counter() - inicio

    print()
    print(f"--> Etapa '{clave}' completada en {duracion:.1f} s")

    return duracion


# ============================================================
# SELECCIÓN DE ETAPAS
# ============================================================

def seleccionar_etapas(argumentos):
    """
    Decide qué etapas se ejecutan a partir de los argumentos.
    """

    if argumentos.solo:

        return [
            etapa
            for etapa in ETAPAS
            if etapa[0] in argumentos.solo
        ]

    seleccionadas = [
        etapa
        for etapa in ETAPAS
        if etapa[3] or argumentos.todo
    ]

    if argumentos.desde:

        claves = [etapa[0] for etapa in seleccionadas]

        if argumentos.desde not in claves:

            raise SystemExit(
                f"La etapa '{argumentos.desde}' no está entre "
                "las seleccionadas."
            )

        seleccionadas = seleccionadas[
            claves.index(argumentos.desde):
        ]

    if argumentos.hasta:

        claves = [etapa[0] for etapa in seleccionadas]

        if argumentos.hasta not in claves:

            raise SystemExit(
                f"La etapa '{argumentos.hasta}' no está entre "
                "las seleccionadas."
            )

        seleccionadas = seleccionadas[
            : claves.index(argumentos.hasta) + 1
        ]

    return seleccionadas


# ============================================================
# ARGUMENTOS
# ============================================================

def leer_argumentos():

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline del TFM: predicción de demanda en "
            "hostelería a partir de meteorología y calendario."
        )
    )

    parser.add_argument(
        "--todo",
        action="store_true",
        help=(
            "Incluye las descargas de AEMET (necesita "
            "conexión y API key)."
        ),
    )

    parser.add_argument(
        "--desde",
        choices=CLAVES,
        help="Empezar en esta etapa.",
    )

    parser.add_argument(
        "--hasta",
        choices=CLAVES,
        help="Terminar en esta etapa.",
    )

    parser.add_argument(
        "--solo",
        nargs="+",
        choices=CLAVES,
        help="Ejecutar únicamente estas etapas.",
    )

    parser.add_argument(
        "--listar",
        action="store_true",
        help="Mostrar las etapas disponibles y salir.",
    )

    parser.add_argument(
        "--continuar",
        action="store_true",
        help=(
            "No detenerse cuando una etapa falla. Útil para "
            "ver todos los errores de una vez."
        ),
    )

    return parser.parse_args()


def listar_etapas():

    print()
    print("ETAPAS DEL PIPELINE")
    print("=" * 70)
    print()

    for clave, titulo, modulo, por_defecto in ETAPAS:

        marca = " " if por_defecto else "*"

        print(f" {marca} {clave:<14} {titulo}")

    print()
    print("* No se ejecuta por defecto (requiere conexión).")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    argumentos = leer_argumentos()

    if argumentos.listar:
        listar_etapas()
        return

    etapas = seleccionar_etapas(argumentos)

    print()
    print("=" * 70)
    print("PIPELINE DEL PROYECTO")
    print("Predicción de demanda en hostelería")
    print("=" * 70)

    print()
    print(f"Etapas a ejecutar: {len(etapas)}")

    for clave, titulo, _, _ in etapas:
        print(f"  - {clave:<14} {titulo}")

    inicio = time.perf_counter()

    completadas = []
    fallidas = []

    for numero, (clave, titulo, modulo, _) in enumerate(
        etapas,
        start=1,
    ):

        try:

            ejecutar_etapa(
                numero,
                len(etapas),
                clave,
                titulo,
                modulo,
            )

            completadas.append(clave)

        except Exception as error:

            fallidas.append((clave, error))

            print()
            print("!" * 70)
            print(f"ERROR EN LA ETAPA '{clave}'")
            print("!" * 70)
            print()
            print(f"{type(error).__name__}: {error}")

            if not argumentos.continuar:

                print()
                print(
                    "El pipeline se detiene. Usa --continuar "
                    "para seguir a pesar de los errores."
                )

                break

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    duracion = time.perf_counter() - inicio

    print()
    print("=" * 70)
    print("RESUMEN DEL PIPELINE")
    print("=" * 70)
    print()

    print(f"Etapas completadas: {len(completadas)}")
    print(f"Etapas con error:   {len(fallidas)}")
    print(f"Tiempo total:       {duracion:.1f} s")

    if fallidas:

        print()
        print("Errores:")

        for clave, error in fallidas:
            print(f"  - {clave}: {error}")

        raise SystemExit(1)

    print()
    print("=" * 70)
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 70)


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
