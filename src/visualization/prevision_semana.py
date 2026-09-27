# src/visualization/prevision_semana.py

# ============================================================
# GRÁFICO DE LA PREVISIÓN DE LOS PRÓXIMOS DÍAS
# ============================================================
#
# Responde de un vistazo a la pregunta con la que el encargado
# abre la aplicación un jueves por la tarde: qué semana viene.
#
# ------------------------------------------------------------
# DECISIONES DEL GRÁFICO
# ------------------------------------------------------------
#
# BARRAS, NO LÍNEA. Una línea uniría los días cerrados con los
# abiertos y sugeriría una continuidad que no existe: el lunes
# no hay "pocos clientes", no hay servicio. Con barras, un día
# cerrado simplemente no tiene barra.
#
# EL INTERVALO, SIEMPRE. La regla del proyecto es que nunca se
# muestra una cifra sola. Cada barra lleva su intervalo del
# 80 %, y como el intervalo escala con la predicción, se ve a
# simple vista que el sábado es más incierto que el miércoles.
#
# LA MEDIA HISTÓRICA COMO REFERENCIA. Un número de clientes no
# dice nada por sí solo: 147 es mucho o poco según el día. El
# guion naranja marca lo que suele traer ese día de la semana,
# así que la comparación que importa —¿este sábado viene más o
# menos de lo normal?— se lee sin hacer cuentas.
#
# UN SOLO EJE. La meteorología que explica la previsión va como
# texto bajo cada barra, no como una segunda escala. Un gráfico
# de dos ejes verticales invita a leer cruces y pendientes que
# solo dependen de cómo se hayan escalado los ejes.
#
# COLORES. #2f6f9f y #e08a2e, los del resto de gráficos del
# proyecto. Comprobados: separación CVD de 23,4 (protanopia) y
# 30,8 (tritanopia), muy por encima del umbral de 8.
#
# ============================================================

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# ESTILO
# ============================================================

COLOR_PREVISION = "#2f6f9f"
COLOR_REFERENCIA = "#e08a2e"
COLOR_INTERVALO = "#7f8c8d"
COLOR_CERRADO = "#e8e8e6"
COLOR_TEXTO = "#2e2620"
COLOR_SUAVE = "#6b6259"


def _texto_meteo(fila):
    """
    Resumen meteorológico de un día, en una línea.
    """

    partes = [f"{fila['tmax']:.0f}° / {fila['tmin']:.0f}°"]

    if fila.get("prob_lluvia", 0) and fila["prob_lluvia"] > 0:
        partes.append(f"{fila['prob_lluvia']:.0f}% lluvia")

    return "\n".join(partes)


# ============================================================
# GRÁFICO
# ============================================================

def grafico_prevision(prevision, titulo=None):
    """
    Construye el gráfico de la previsión semanal.

    Recibe el DataFrame que devuelve
    src/models/prediccion_manana.prevision_proximos_dias() y
    devuelve la figura de matplotlib, sin guardarla: quien la
    llama decide si la muestra en pantalla o la escribe en
    disco.
    """

    datos = prevision.copy().reset_index(drop=True)

    posiciones = range(len(datos))

    abiertos = datos["abierto"] == 1

    figura, eje = plt.subplots(
        figsize=(max(8, 1.5 * len(datos)), 4.8),
    )

    # --------------------------------------------------------
    # DÍAS DE CIERRE
    # --------------------------------------------------------
    #
    # Se marcan con una franja de fondo, no con una barra a
    # cero: un día cerrado no es un día de cero clientes, es un
    # día sin servicio. Es la misma distinción que hace la capa
    # gold al excluirlos del modelado.

    for posicion, fila in datos[~abiertos].iterrows():

        eje.axvspan(
            posicion - 0.5,
            posicion + 0.5,
            color=COLOR_CERRADO,
            zorder=0,
        )

        eje.text(
            posicion,
            0.5,
            "CERRADO",
            ha="center",
            va="center",
            fontsize=9,
            color=COLOR_SUAVE,
            rotation=90,
            transform=eje.get_xaxis_transform(),
            zorder=2,
        )

    # --------------------------------------------------------
    # BARRAS E INTERVALOS
    # --------------------------------------------------------

    visibles = datos[abiertos]

    if not visibles.empty:

        errores = [
            (
                visibles["clientes_estimados"]
                - visibles["intervalo_inferior"]
            ).values,
            (
                visibles["intervalo_superior"]
                - visibles["clientes_estimados"]
            ).values,
        ]

        eje.bar(
            visibles.index,
            visibles["clientes_estimados"],
            width=0.62,
            color=COLOR_PREVISION,
            edgecolor="white",
            linewidth=2,
            zorder=3,
            label="Clientes estimados",
        )

        eje.errorbar(
            visibles.index,
            visibles["clientes_estimados"],
            yerr=errores,
            fmt="none",
            ecolor=COLOR_INTERVALO,
            elinewidth=1.4,
            capsize=5,
            capthick=1.4,
            zorder=4,
        )

    # --------------------------------------------------------
    # REFERENCIA HISTÓRICA
    # --------------------------------------------------------

    referencias = datos[
        abiertos & datos["media_historica"].notna()
    ]

    for posicion, fila in referencias.iterrows():

        eje.plot(
            [posicion - 0.34, posicion + 0.34],
            [fila["media_historica"]] * 2,
            color=COLOR_REFERENCIA,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=5,
            label=(
                "Media histórica de ese día"
                if posicion == referencias.index[0]
                else None
            ),
        )

    # --------------------------------------------------------
    # ETIQUETAS DIRECTAS
    # --------------------------------------------------------
    #
    # El número encima de la barra y la plantilla dentro. Son
    # las dos cifras que el encargado necesita, y así no hay
    # que leer el eje para nada.

    for posicion, fila in visibles.iterrows():

        eje.text(
            posicion,
            fila["intervalo_superior"] + 4,
            f"{fila['clientes_estimados']:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=COLOR_TEXTO,
            zorder=6,
        )

        eje.text(
            posicion,
            4,
            f"{int(fila['empleados_recomendados'])} emp.",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    # --------------------------------------------------------
    # EJES
    # --------------------------------------------------------

    etiquetas = [
        f"{fila['dia'][:3]} {pd.to_datetime(fila['fecha']):%d/%m}"
        f"\n{_texto_meteo(fila)}"
        for _, fila in datos.iterrows()
    ]

    eje.set_xticks(list(posiciones))
    eje.set_xticklabels(etiquetas, fontsize=9.5)

    eje.set_ylabel("Clientes", fontsize=10, color=COLOR_SUAVE)

    # Techo del eje. En agosto el local cierra el mes entero,
    # así que puede no haber ninguna barra: entonces todas las
    # columnas son nulas, su máximo es NaN, y NaN pasa el
    # `or` porque es un valor verdadero. Hay que comprobarlo
    # con isna, no con una condición implícita.

    candidatos = [
        datos["intervalo_superior"].max(),
        datos["media_historica"].max(),
    ]

    validos = [
        valor for valor in candidatos if not pd.isna(valor)
    ]

    techo = max(validos) if validos else 100

    # Aire arriba para las etiquetas y la leyenda.
    eje.set_ylim(0, techo * 1.30)

    eje.set_xlim(-0.6, len(datos) - 0.4)

    eje.grid(axis="y", color="#e8e8e8", linewidth=1, zorder=1)
    eje.set_axisbelow(True)

    for lado in ("top", "right", "left"):
        eje.spines[lado].set_visible(False)

    eje.spines["bottom"].set_color("#dddddd")

    eje.tick_params(
        axis="both",
        length=0,
        colors=COLOR_SUAVE,
    )

    # --------------------------------------------------------
    # TÍTULO Y LEYENDA
    # --------------------------------------------------------

    if titulo:
        eje.set_title(
            titulo,
            fontsize=13,
            fontweight="bold",
            color=COLOR_TEXTO,
            loc="left",
            pad=34,
        )

    # La serie principal primero: matplotlib ordena por el
    # momento en que se dibuja cada artista, y los guiones de
    # referencia se pintan después de las barras.

    manejadores, textos = eje.get_legend_handles_labels()

    orden = sorted(
        range(len(textos)),
        key=lambda i: textos[i] != "Clientes estimados",
    )

    eje.legend(
        [manejadores[i] for i in orden],
        [textos[i] for i in orden],
        loc="lower left",
        frameon=False,
        fontsize=9.5,
        ncols=2,
        bbox_to_anchor=(0, 1.0),
        borderaxespad=0.2,
    )

    figura.tight_layout()

    return figura


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

def main():
    """
    Genera el gráfico y lo guarda, para poder revisarlo sin
    levantar la aplicación.
    """

    from src.config import ruta_grafico

    from src.models.prediccion_manana import (
        prevision_proximos_dias,
    )

    prevision, info = prevision_proximos_dias()

    if prevision is None:

        raise SystemExit(
            "No hay previsión utilizable. Descarga la actual:\n"
            "    python -m src.extraction.descargar_prediccion"
        )

    figura = grafico_prevision(
        prevision,
        titulo="Previsión de clientes para los próximos días",
    )

    ruta = ruta_grafico("prevision_semana")

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)

    print(f"Municipio:  {info['municipio']}")
    print(f"Elaborada:  {info['elaborado']}")
    print(f"Guardado en: {ruta}")


if __name__ == "__main__":
    main()
