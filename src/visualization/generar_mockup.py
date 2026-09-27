# src/visualization/generar_mockup.py

# ============================================================
# MOCKUP DEL FRONTAL
# ============================================================
#
# Genera la imagen de la pantalla principal de la aplicación
# que se incluye en la Entrega 5.
#
# El mockup NO está dibujado a mano: se construye por código y
# las cifras que muestra son las que devuelve de verdad el
# modelo entrenado para ese día y esas condiciones. Así la
# imagen de la memoria y el comportamiento real de la
# aplicación no pueden separarse.
#
# Salida:
#
#     docs/assets/05_mockup_frontal.png
#
# ============================================================

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    ASSETS_DIR,
    GOLD_DATASET,
    LOCALIDAD,
    NOMBRE_RESTAURANTE,
)

from src.models.prediccion import (
    cargar_calendario,
    cargar_modelo_produccion,
    contexto_historico,
    evaluar_plantilla,
    explicar,
    predecir,
)


# ============================================================
# ESCENARIO REPRESENTADO
# ============================================================
#
# Un sábado de junio con buen tiempo: el caso de mayor
# afluencia y, por tanto, el que más se juega el encargado.

FECHA_MOCKUP = date(2026, 6, 20)

TMAX_MOCKUP = 28.0
TMIN_MOCKUP = 15.0
PREC_MOCKUP = 0.0


# ============================================================
# PALETA
# ============================================================

FONDO = "#f5f6f8"
PANEL = "#ffffff"
LATERAL = "#eaecf0"
TINTA = "#1b2733"
SUAVE = "#5a6b7b"
ACENTO = "#1f4e79"
VERDE = "#27ae60"
VERDE_FONDO = "#eaf4ec"
AMBAR = "#d68910"
AMBAR_FONDO = "#fdf6e3"
AZUL_FONDO = "#e8f0f7"


# ============================================================
# UTILIDADES DE DIBUJO
# ============================================================

def caja(eje, x, y, ancho, alto, color, borde=None, radio=0.012):
    """
    Dibuja un rectángulo con esquinas redondeadas.
    """

    rectangulo = patches.FancyBboxPatch(
        (x, y),
        ancho,
        alto,
        boxstyle=f"round,pad=0,rounding_size={radio}",
        linewidth=1.2 if borde else 0,
        edgecolor=borde or "none",
        facecolor=color,
    )

    eje.add_patch(rectangulo)

    return rectangulo


def barra_lateral_color(eje, x, y, alto, color, ancho=0.006):
    """
    Franja de color a la izquierda de un bloque.
    """

    eje.add_patch(
        patches.Rectangle(
            (x, y),
            ancho,
            alto,
            facecolor=color,
            edgecolor="none",
        )
    )


def texto(
    eje,
    x,
    y,
    contenido,
    tamano=11,
    color=TINTA,
    peso="normal",
    ha="left",
    va="center",
):

    eje.text(
        x,
        y,
        contenido,
        fontsize=tamano,
        color=color,
        fontweight=peso,
        ha=ha,
        va=va,
        family="DejaVu Sans",
    )


def control(eje, x, y, ancho, etiqueta, valor):
    """
    Representa un control de entrada del panel lateral.
    """

    texto(eje, x, y + 0.028, etiqueta, 9.5, SUAVE)

    caja(eje, x, y - 0.012, ancho, 0.036, PANEL, "#c9d1d9", 0.008)

    texto(eje, x + 0.012, y + 0.006, valor, 10.5, TINTA, "bold")


# ============================================================
# CONSTRUCCIÓN DEL MOCKUP
# ============================================================

def generar_mockup(resultado, historico, motivos):
    """
    Dibuja la pantalla principal de la aplicación.
    """

    figura = plt.figure(figsize=(16, 10), dpi=110)

    eje = figura.add_axes([0, 0, 1, 1])

    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1)

    eje.axis("off")

    eje.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=FONDO,
            edgecolor="none",
        )
    )

    # ========================================================
    # PANEL LATERAL
    # ========================================================

    eje.add_patch(
        patches.Rectangle(
            (0, 0),
            0.235,
            1,
            facecolor=LATERAL,
            edgecolor="none",
        )
    )

    texto(eje, 0.028, 0.955, "Día a planificar", 13, TINTA, "bold")

    control(
        eje,
        0.028,
        0.888,
        0.178,
        "Fecha",
        resultado["fecha"].strftime("%d/%m/%Y"),
    )

    eje.plot(
        [0.028, 0.206],
        [0.852, 0.852],
        color="#c9d1d9",
        linewidth=1,
    )

    texto(
        eje,
        0.028,
        0.822,
        "Previsión meteorológica",
        12,
        TINTA,
        "bold",
    )

    texto(
        eje,
        0.028,
        0.795,
        "Previsión de AEMET para ese día",
        8.5,
        SUAVE,
    )

    control(
        eje,
        0.028,
        0.735,
        0.178,
        "Temperatura máxima (°C)",
        f"{resultado['tmax']:.1f}",
    )

    control(
        eje,
        0.028,
        0.655,
        0.178,
        "Temperatura mínima (°C)",
        f"{resultado['tmin']:.1f}",
    )

    control(
        eje,
        0.028,
        0.575,
        0.178,
        "Precipitación prevista (mm)",
        f"{resultado['prec']:.1f}",
    )

    eje.plot(
        [0.028, 0.206],
        [0.535, 0.535],
        color="#c9d1d9",
        linewidth=1,
    )

    # --- Casilla de festivo ---

    caja(eje, 0.028, 0.487, 0.018, 0.024, PANEL, "#c9d1d9", 0.005)

    texto(eje, 0.055, 0.499, "Marcar como festivo", 10, TINTA)

    texto(
        eje,
        0.028,
        0.455,
        "El calendario ya incluye los\n"
        "festivos de Arganda del Rey.",
        8.5,
        SUAVE,
        va="top",
    )

    # --- Pie del panel ---

    texto(
        eje,
        0.028,
        0.055,
        "Datos meteorológicos\nAEMET OpenData · 3182Y",
        8.5,
        SUAVE,
        va="center",
    )

    # ========================================================
    # CABECERA
    # ========================================================

    texto(
        eje,
        0.265,
        0.955,
        "Planificador de servicio",
        21,
        TINTA,
        "bold",
    )

    texto(
        eje,
        0.265,
        0.918,
        f"{NOMBRE_RESTAURANTE} · {LOCALIDAD} (Madrid)   ·   "
        "Estima cuántos clientes vendrán y con cuánta "
        "plantilla cubrir el servicio.",
        10.5,
        SUAVE,
    )

    # --- Contexto del día ---

    etiquetas = [
        resultado["fecha"].strftime("%d/%m/%Y"),
        resultado["nombre_dia"],
    ]

    periodo = resultado.get("periodo_vacacional") or ""

    if periodo and periodo.lower() != "ninguna":
        etiquetas.append(periodo)

    texto(
        eje,
        0.265,
        0.872,
        "   ·   ".join(etiquetas),
        13.5,
        TINTA,
        "bold",
    )

    # ========================================================
    # BLOQUE 1 · CLIENTES ESTIMADOS
    # ========================================================

    caja(eje, 0.265, 0.632, 0.42, 0.212, AZUL_FONDO, radio=0.012)

    barra_lateral_color(eje, 0.265, 0.632, 0.212, ACENTO)

    texto(eje, 0.288, 0.812, "Clientes estimados", 11, SUAVE)

    texto(
        eje,
        0.288,
        0.746,
        f"{resultado['prediccion']:.0f}",
        58,
        ACENTO,
        "bold",
    )

    texto(
        eje,
        0.288,
        0.688,
        "Rango probable (8 de cada 10 días):  "
        f"{resultado['intervalo_inferior']:.0f} – "
        f"{resultado['intervalo_superior']:.0f}",
        11,
        TINTA,
    )

    if historico:

        referencia = (
            historico.get("media_dia_semana_mes")
            or historico["media_dia_semana"]
        )

        diferencia = resultado["prediccion"] - referencia

        texto(
            eje,
            0.288,
            0.658,
            f"Un {resultado['nombre_dia'].lower()} de junio "
            f"suele traer {referencia:.0f} clientes. "
            f"La previsión está {abs(diferencia):.0f} "
            f"{'por encima' if diferencia >= 0 else 'por debajo'}.",
            10,
            SUAVE,
        )

    # ========================================================
    # BLOQUE 2 · PLANTILLA
    # ========================================================

    empleados = resultado["empleados_recomendados"]

    estado, titulo, detalle = evaluar_plantilla(
        resultado["prediccion"],
        empleados,
    )

    color_estado = {
        "riesgo": ("#fdece9", "#c0392b"),
        "ajustado": (AMBAR_FONDO, AMBAR),
        "holgado": (VERDE_FONDO, VERDE),
    }[estado]

    texto(eje, 0.705, 0.822, "Plantilla prevista", 11, SUAVE)

    caja(eje, 0.705, 0.772, 0.115, 0.038, PANEL, "#c9d1d9", 0.008)

    texto(
        eje,
        0.718,
        0.791,
        f"{empleados}",
        13,
        TINTA,
        "bold",
    )

    texto(eje, 0.755, 0.791, "empleados", 10.5, SUAVE)

    caja(
        eje,
        0.705,
        0.632,
        0.27,
        0.122,
        color_estado[0],
        radio=0.012,
    )

    barra_lateral_color(
        eje,
        0.705,
        0.632,
        0.122,
        color_estado[1],
    )

    texto(
        eje,
        0.728,
        0.726,
        f"Con {empleados} empleados",
        10.5,
        SUAVE,
    )

    texto(
        eje,
        0.728,
        0.690,
        titulo,
        17,
        color_estado[1],
        "bold",
    )

    texto(eje, 0.728, 0.655, detalle, 10.5, TINTA)

    # ========================================================
    # AVISO DE ESCENARIO ALTO
    # ========================================================

    caja(eje, 0.265, 0.565, 0.71, 0.05, "#e8f0f7", radio=0.01)

    barra_lateral_color(eje, 0.265, 0.565, 0.05, ACENTO)

    from src.models.prediccion import recomendar_empleados

    empleados_extra = max(
        0,
        recomendar_empleados(resultado["intervalo_superior"])
        - empleados,
    )

    if empleados_extra > 0:

        aviso = (
            "Si el día se va al extremo alto del rango "
            f"({resultado['intervalo_superior']:.0f} clientes), "
            f"harían falta {empleados_extra} empleados más. "
            "Conviene tener a alguien localizable."
        )

    else:

        ratio_alto = (
            resultado["intervalo_superior"] / empleados
        )

        aviso = (
            "Si el día se va al extremo alto del rango "
            f"({resultado['intervalo_superior']:.0f} clientes), "
            f"la plantilla quedaría a {ratio_alto:.0f} clientes "
            "por empleado. Es el máximo que cubre el local."
        )

    texto(eje, 0.288, 0.590, aviso, 10.5, TINTA)

    # ========================================================
    # BLOQUE 3 · POR QUÉ
    # ========================================================

    caja(eje, 0.265, 0.285, 0.42, 0.255, PANEL, "#e3e7eb")

    texto(
        eje,
        0.288,
        0.508,
        "Por qué esta previsión",
        13,
        TINTA,
        "bold",
    )

    posicion = 0.468

    for motivo in motivos[:4]:

        # Se parte el texto en dos líneas si es largo.
        if len(motivo) > 62:

            corte = motivo.rfind(" ", 0, 62)

            lineas = [motivo[:corte], motivo[corte + 1:]]

        else:

            lineas = [motivo]

        texto(eje, 0.290, posicion, "•", 11, ACENTO)

        for indice, linea in enumerate(lineas):

            texto(
                eje,
                0.302,
                posicion - indice * 0.024,
                linea,
                10.5,
                TINTA,
            )

        posicion -= 0.024 * len(lineas) + 0.014

    texto(
        eje,
        0.288,
        0.308,
        "Construido con los valores reales que recibe el "
        "modelo.\nNo interviene IA generativa: no puede "
        "inventar una causa.",
        8.8,
        SUAVE,
        va="center",
    )

    # ========================================================
    # BLOQUE 4 · CONDICIONES DEL DÍA
    # ========================================================

    caja(eje, 0.705, 0.285, 0.27, 0.255, PANEL, "#e3e7eb")

    texto(
        eje,
        0.728,
        0.508,
        "Condiciones del día",
        13,
        TINTA,
        "bold",
    )

    condiciones = [
        ("Temperatura máxima", f"{resultado['tmax']:.1f} °C"),
        ("Temperatura mínima", f"{resultado['tmin']:.1f} °C"),
        ("Temperatura media", f"{resultado['tmed']:.1f} °C"),
        ("Precipitación", f"{resultado['prec']:.1f} mm"),
        ("Día de la semana", resultado["nombre_dia"]),
        (
            "Festivo",
            "Sí" if resultado["es_festivo"] else "No",
        ),
    ]

    posicion = 0.468

    for indice, (nombre, valor) in enumerate(condiciones):

        if indice % 2 == 0:

            caja(
                eje,
                0.716,
                posicion - 0.014,
                0.248,
                0.028,
                "#f7f8fa",
                radio=0.004,
            )

        texto(eje, 0.728, posicion, nombre, 10, SUAVE)

        texto(
            eje,
            0.952,
            posicion,
            valor,
            10,
            TINTA,
            "bold",
            ha="right",
        )

        posicion -= 0.032

    # ========================================================
    # BLOQUE 5 · FIABILIDAD
    # ========================================================

    texto(
        eje,
        0.265,
        0.243,
        "Hasta dónde fiarse de esta cifra",
        13,
        TINTA,
        "bold",
    )

    metricas = resultado["_metricas"]

    tarjetas = [
        (
            "Error medio",
            f"{metricas['mae']:.1f}",
            "clientes",
        ),
        (
            "Error relativo",
            f"{metricas['mape']:.1f} %",
            "sobre la demanda real",
        ),
        (
            "Variabilidad explicada",
            f"{metricas['r2']:.0%}",
            "de la variación diaria",
        ),
        (
            "Modelo",
            "Regresión lineal",
            f"{resultado['_registros']} días de historia",
        ),
    ]

    x = 0.265

    for etiqueta, valor, detalle_tarjeta in tarjetas:

        caja(eje, x, 0.128, 0.166, 0.098, PANEL, "#e3e7eb")

        texto(eje, x + 0.014, 0.206, etiqueta, 9.5, SUAVE)

        texto(
            eje,
            x + 0.014,
            0.176,
            valor,
            17,
            TINTA,
            "bold",
        )

        texto(
            eje,
            x + 0.014,
            0.148,
            detalle_tarjeta,
            8.5,
            SUAVE,
        )

        x += 0.182

    # ========================================================
    # AVISO DE LIMITACIÓN
    # ========================================================

    caja(eje, 0.265, 0.038, 0.71, 0.075, AMBAR_FONDO, radio=0.01)

    barra_lateral_color(eje, 0.265, 0.038, 0.075, AMBAR)

    texto(
        eje,
        0.288,
        0.096,
        "Limitación conocida",
        10.5,
        AMBAR,
        "bold",
    )

    texto(
        eje,
        0.288,
        0.068,
        f"El modelo se queda corto una media de "
        f"{resultado['_sesgo']:.0f} clientes porque el negocio "
        "crece más deprisa de lo que sabe extrapolar.",
        9.5,
        TINTA,
    )

    texto(
        eje,
        0.288,
        0.050,
        "Si la decisión es ajustada, tira hacia el extremo "
        "alto del rango. Se corrige reentrenando cada pocos "
        "meses.",
        9.5,
        TINTA,
    )

    return figura


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERACIÓN DEL MOCKUP DEL FRONTAL")
    print("=" * 60)

    modelo, metadatos = cargar_modelo_produccion()

    calendario = cargar_calendario()

    gold = (
        pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])
        if GOLD_DATASET.exists()
        else None
    )

    resultado = predecir(
        fecha=FECHA_MOCKUP,
        tmax=TMAX_MOCKUP,
        tmin=TMIN_MOCKUP,
        prec=PREC_MOCKUP,
        modelo=modelo,
        metadatos=metadatos,
        calendario=calendario,
    )

    historico = contexto_historico(FECHA_MOCKUP, gold)

    motivos = explicar(resultado, historico)

    # Datos adicionales que muestra la pantalla.
    resultado["_metricas"] = metadatos["metricas_test"]
    resultado["_registros"] = metadatos["registros_entrenamiento"]
    resultado["_sesgo"] = metadatos.get("sesgo_test", 0)

    print()
    print(f"Escenario: {FECHA_MOCKUP} · {resultado['nombre_dia']}")

    print(
        f"Predicción: {resultado['prediccion']:.0f} clientes "
        f"({resultado['intervalo_inferior']:.0f} - "
        f"{resultado['intervalo_superior']:.0f})"
    )

    print(
        f"Plantilla:  {resultado['empleados_recomendados']} "
        "empleados"
    )

    figura = generar_mockup(resultado, historico, motivos)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    ruta = ASSETS_DIR / "05_mockup_frontal.png"

    figura.savefig(
        ruta,
        dpi=110,
        facecolor=FONDO,
    )

    plt.close(figura)

    print()
    print(f"Mockup guardado en: {ruta}")

    print()
    print("=" * 60)
    print("MOCKUP GENERADO CORRECTAMENTE")
    print("=" * 60)

    return ruta


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
