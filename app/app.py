# app/app.py

# ============================================================
# APLICACIÓN WEB · PLANIFICADOR DE SERVICIO
# ============================================================
#
# Frontal del proyecto. Está pensado para una persona
# concreta: el encargado del restaurante, decidiendo el jueves
# por la tarde a quién llama para el fin de semana.
#
# Principios de diseño:
#
#   1. La pantalla responde a UNA pregunta: cuánta gente viene
#      y con cuánta plantilla lo cubro.
#
#   2. Nunca se muestra una cifra sola. Toda predicción va
#      acompañada de su intervalo, de la referencia histórica
#      y de una explicación de por qué sale ese número.
#
#   3. El usuario manda. La plantilla recomendada es una
#      propuesta que puede ajustar, y la aplicación le dice al
#      instante cómo quedaría la jornada con su decisión.
#
#   4. Nada está escrito a mano. Las métricas que aparecen se
#      leen de los metadatos del modelo entrenado, así que no
#      pueden quedar desfasadas.
#
# Ejecución:
#
#     streamlit run app/app.py
#
# ============================================================

import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    COMPARACION_MODELOS,
    EMPLEADOS_MAXIMO,
    EMPLEADOS_MINIMO,
    GOLD_DATASET,
    LOCALIDAD,
    NOMBRE_RESTAURANTE,
    ruta_grafico,
)

from src.models.prediccion import (
    cargar_calendario,
    cargar_modelo_produccion,
    contexto_historico,
    evaluar_plantilla,
    explicar,
    predecir,
    recomendar_empleados,
)

from src.models.prediccion_manana import (
    prevision_proximos_dias,
)

from src.visualization.prevision_semana import (
    grafico_prevision,
)


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Planificador de servicio",
    page_icon="🍽️",
    layout="wide",
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .bloque-resultado {
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 8px;
    }
    .bloque-riesgo   { background: #fdece9; border-left: 6px solid #c0392b; }
    .bloque-ajustado { background: #fdf6e3; border-left: 6px solid #d68910; }
    .bloque-holgado  { background: #eaf4ec; border-left: 6px solid #27ae60; }
    .cifra-principal { font-size: 3.2rem; font-weight: 700; line-height: 1.1; }
    .etiqueta        { font-size: 0.95rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE RECURSOS
# ============================================================

@st.cache_resource(show_spinner=False)
def cargar_recursos():
    """
    Carga el modelo, sus metadatos y los datos de referencia.

    Se cachean para que la aplicación responda al instante en
    cada cambio de la interfaz.
    """

    modelo, metadatos = cargar_modelo_produccion()

    calendario = cargar_calendario()

    gold = (
        pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])
        if GOLD_DATASET.exists()
        else None
    )

    comparacion = (
        pd.read_csv(COMPARACION_MODELOS)
        if COMPARACION_MODELOS.exists()
        else None
    )

    return modelo, metadatos, calendario, gold, comparacion


try:

    (
        modelo,
        metadatos,
        calendario,
        gold,
        comparacion,
    ) = cargar_recursos()

except FileNotFoundError as error:

    st.error(
        "No se encuentra el modelo entrenado.\n\n"
        "Ejecuta primero el pipeline del proyecto:\n\n"
        "```\npython pipeline.py\n```"
    )

    st.caption(str(error))

    st.stop()


# ============================================================
# CABECERA
# ============================================================

st.title("🍽️ Planificador de servicio")

st.markdown(
    f"**{NOMBRE_RESTAURANTE}** · {LOCALIDAD} (Madrid)  \n"
    "Estima cuántos clientes vendrán y con cuánta plantilla "
    "cubrir el servicio."
)


# ============================================================
# BARRA LATERAL · ENTRADAS
# ============================================================

st.sidebar.header("Día a planificar")

fecha = st.sidebar.date_input(
    "Fecha",
    value=date.today() + timedelta(days=1),
    help=(
        "El sistema está pensado para planificar con uno o "
        "varios días de antelación."
    ),
)

st.sidebar.divider()

st.sidebar.subheader("Previsión meteorológica")

st.sidebar.caption(
    "Introduce la previsión de AEMET para ese día, no el "
    "tiempo que hace hoy."
)

temperatura_maxima = st.sidebar.slider(
    "Temperatura máxima (°C)",
    min_value=-5.0,
    max_value=48.0,
    value=24.0,
    step=0.5,
)

temperatura_minima = st.sidebar.slider(
    "Temperatura mínima (°C)",
    min_value=-15.0,
    max_value=35.0,
    value=12.0,
    step=0.5,
)

if temperatura_minima > temperatura_maxima:

    st.sidebar.error(
        "La temperatura mínima no puede superar a la máxima."
    )

    st.stop()

precipitacion = st.sidebar.slider(
    "Precipitación prevista (mm)",
    min_value=0.0,
    max_value=60.0,
    value=0.0,
    step=0.5,
    help=(
        "Si solo tienes la probabilidad de lluvia, una regla "
        "razonable es multiplicarla por 6 mm, que es lo que "
        "llueve de media un día lluvioso aquí."
    ),
)

st.sidebar.divider()

forzar_festivo = st.sidebar.checkbox(
    "Marcar como festivo",
    value=False,
    help=(
        "El calendario del proyecto ya incluye los festivos "
        "nacionales, de la Comunidad de Madrid y locales de "
        "Arganda del Rey. Marca esta casilla solo si sabes de "
        "un festivo o evento que el calendario no recoge."
    ),
)


# ============================================================
# PREDICCIÓN
# ============================================================

resultado = predecir(
    fecha=fecha,
    tmax=temperatura_maxima,
    tmin=temperatura_minima,
    prec=precipitacion,
    modelo=modelo,
    metadatos=metadatos,
    calendario=calendario,
    es_festivo=True if forzar_festivo else None,
)

historico = contexto_historico(fecha, gold)


# ============================================================
# DÍA DE CIERRE
# ============================================================

if resultado["cerrado"] and not forzar_festivo:

    st.warning(
        f"**{fecha.strftime('%d/%m/%Y')} · "
        f"{resultado['nombre_dia']}** — el restaurante está "
        f"cerrado ese día ({resultado['motivo_cierre']}).\n\n"
        "No hay servicio que planificar. Elige otra fecha en "
        "el panel de la izquierda."
    )

    st.stop()


# ============================================================
# CONTEXTO DEL DÍA
# ============================================================

etiquetas = [f"**{resultado['nombre_dia']}**"]

if resultado["es_festivo"]:

    etiquetas.append(
        f"Festivo: {resultado['nombre_festivo']}"
        if resultado["nombre_festivo"]
        else "Festivo"
    )

periodo = resultado["periodo_vacacional"]

if periodo and periodo.lower() != "ninguna":
    etiquetas.append(periodo)

st.markdown(
    f"### {fecha.strftime('%d/%m/%Y')} · "
    + "  ·  ".join(etiquetas)
)


# ============================================================
# RESULTADO PRINCIPAL
# ============================================================

columna_prediccion, columna_plantilla = st.columns([3, 2])


with columna_prediccion:

    st.markdown(
        f"""
        <div class="bloque-resultado bloque-holgado">
          <div class="etiqueta">Clientes estimados</div>
          <div class="cifra-principal">
            {resultado['prediccion']:.0f}
          </div>
          <div class="etiqueta">
            Rango probable (8 de cada 10 días):
            <b>{resultado['intervalo_inferior']:.0f} –
            {resultado['intervalo_superior']:.0f}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if historico:

        referencia = (
            historico["media_dia_semana_mes"]
            or historico["media_dia_semana"]
        )

        diferencia = resultado["prediccion"] - referencia

        st.caption(
            f"Un {resultado['nombre_dia'].lower()} de "
            f"{fecha.strftime('%B').lower()} suele traer "
            f"**{referencia:.0f}** clientes. "
            f"La previsión está **{abs(diferencia):.0f} "
            f"{'por encima' if diferencia >= 0 else 'por debajo'}**."
        )


with columna_plantilla:

    recomendados = resultado["empleados_recomendados"]

    empleados = st.number_input(
        "Plantilla prevista",
        min_value=EMPLEADOS_MINIMO,
        max_value=EMPLEADOS_MAXIMO,
        value=recomendados,
        step=1,
        help=(
            "Empieza por la recomendación y ajústala si tienes "
            "restricciones de plantilla."
        ),
    )

    estado, titulo, detalle = evaluar_plantilla(
        resultado["prediccion"],
        empleados,
    )

    st.markdown(
        f"""
        <div class="bloque-resultado bloque-{estado}">
          <div class="etiqueta">Con {empleados} empleados</div>
          <div style="font-size:1.5rem;font-weight:700;
                      margin:4px 0;">
            {titulo}
          </div>
          <div class="etiqueta">{detalle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if empleados != recomendados:

        st.caption(
            f"La recomendación del sistema son "
            f"**{recomendados} empleados**."
        )

    else:

        st.caption("Es la plantilla recomendada.")


# ============================================================
# ESCENARIO PESIMISTA
# ============================================================
#
# Planificar solo con la predicción puntual deja al negocio
# expuesto la mitad de los días. Se muestra también qué haría
# falta si el día se va al extremo alto del intervalo.

plantilla_maxima = recomendar_empleados(
    resultado["intervalo_superior"]
)

if plantilla_maxima > empleados:

    st.info(
        f"Si el día se va al extremo alto del rango "
        f"({resultado['intervalo_superior']:.0f} clientes), "
        f"harían falta **{plantilla_maxima} empleados**. "
        "Conviene tener a alguien localizable."
    )


# ============================================================
# EXPLICACIÓN
# ============================================================

st.divider()

columna_motivos, columna_condiciones = st.columns([3, 2])


with columna_motivos:

    st.subheader("Por qué esta previsión")

    for motivo in explicar(resultado, historico):
        st.markdown(f"- {motivo}")

    st.caption(
        "La explicación se construye con los valores reales "
        "que ha recibido el modelo. No interviene ninguna IA "
        "generativa, así que no puede inventar una causa que "
        "no esté en los datos."
    )


with columna_condiciones:

    st.subheader("Condiciones del día")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Variable": "Temperatura máxima",
                    "Valor": f"{temperatura_maxima:.1f} °C",
                },
                {
                    "Variable": "Temperatura mínima",
                    "Valor": f"{temperatura_minima:.1f} °C",
                },
                {
                    "Variable": "Temperatura media",
                    "Valor": f"{resultado['tmed']:.1f} °C",
                },
                {
                    "Variable": "Precipitación",
                    "Valor": f"{precipitacion:.1f} mm",
                },
                {
                    "Variable": "Día de la semana",
                    "Valor": resultado["nombre_dia"],
                },
                {
                    "Variable": "Festivo",
                    "Valor": (
                        "Sí" if resultado["es_festivo"] else "No"
                    ),
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# LOS PRÓXIMOS DÍAS
# ============================================================
#
# La pantalla de arriba responde por un día concreto, con la
# meteorología que el usuario teclea. Esta sección responde por
# la semana entera con la previsión real descargada de AEMET,
# que es como se decide la plantilla: no día a día, sino viendo
# el fin de semana completo.

st.divider()


@st.cache_data(show_spinner=False, ttl=900)
def cargar_prevision_semana():
    """
    Previsión de los próximos días a partir del último fichero
    descargado de AEMET.

    Se cachea quince minutos: el cálculo es rápido, pero se
    repetiría en cada movimiento de un control de la barra
    lateral y la pantalla daría tirones.

    Los recursos ya cargados se pasan a la función para no
    volver a leer el modelo ni el histórico de disco.
    """

    return prevision_proximos_dias(
        modelo=modelo,
        metadatos=metadatos,
        calendario=calendario,
        gold=gold,
    )


st.subheader("Los próximos días")

try:
    prevision, info = cargar_prevision_semana()

except FileNotFoundError:
    prevision, info = None, None


if prevision is None:

    st.info(
        "No hay ninguna previsión meteorológica descargada. "
        "Para ver la semana completa:\n\n"
        "```\npython -m src.extraction.descargar_prediccion\n```"
    )

else:

    st.caption(
        f"Previsión de AEMET para {info['municipio']} "
        f"({info['provincia']}), elaborada el "
        f"{pd.to_datetime(info['elaborado']):%d/%m/%Y a las %H:%M}."
    )

    if info["caducada"]:

        st.warning(
            "**La previsión descargada ha caducado.** AEMET "
            "publica solo siete días y estos ya han pasado. "
            "Se muestran igualmente para poder revisar el "
            "sistema, pero no sirven para planificar. "
            "Descarga la actual con "
            "`python -m src.extraction.descargar_prediccion`."
        )

    figura = grafico_prevision(prevision)

    st.pyplot(figura, use_container_width=True)

    plt.close(figura)

    # ----------------------------------------------------
    # LECTURA DEL GRÁFICO
    # ----------------------------------------------------
    #
    # El mismo principio que en el resto de la aplicación: la
    # cifra no se deja sola. Se señala el día de más carga y
    # los que se desvían de lo normal, que es donde el
    # encargado tiene que decidir algo.

    abiertos = prevision[prevision["abierto"] == 1]

    if not abiertos.empty:

        pico = abiertos.loc[
            abiertos["clientes_estimados"].idxmax()
        ]

        mensajes = [
            f"El día más cargado es el **{pico['dia'].lower()} "
            f"{pd.to_datetime(pico['fecha']):%d/%m}**, con "
            f"**{pico['clientes_estimados']:.0f} clientes** "
            f"estimados y "
            f"**{int(pico['empleados_recomendados'])} "
            f"empleados** recomendados."
        ]

        con_referencia = abiertos[
            abiertos["media_historica"].notna()
        ].copy()

        if not con_referencia.empty:

            con_referencia["desvio"] = (
                con_referencia["clientes_estimados"]
                - con_referencia["media_historica"]
            )

            destacado = con_referencia.loc[
                con_referencia["desvio"].abs().idxmax()
            ]

            if abs(destacado["desvio"]) >= 10:

                direccion = (
                    "por encima"
                    if destacado["desvio"] > 0
                    else "por debajo"
                )

                mensajes.append(
                    f"El que más se aparta de lo normal es el "
                    f"**{destacado['dia'].lower()} "
                    f"{pd.to_datetime(destacado['fecha']):%d/%m}**: "
                    f"{abs(destacado['desvio']):.0f} clientes "
                    f"{direccion} de lo que suele traer ese día."
                )

            else:

                mensajes.append(
                    "Ningún día se aparta más de 10 clientes "
                    "de su media histórica: es una semana "
                    "dentro de lo habitual."
                )

        for mensaje in mensajes:
            st.markdown(f"- {mensaje}")

    with st.expander("Ver la previsión en tabla"):

        tabla = prevision.copy()

        tabla["fecha"] = pd.to_datetime(
            tabla["fecha"]
        ).dt.strftime("%d/%m/%Y")

        tabla["abierto"] = tabla["abierto"].map(
            {1: "Sí", 0: "No"}
        )

        st.dataframe(
            tabla[
                [
                    "fecha",
                    "dia",
                    "abierto",
                    "motivo_cierre",
                    "tmax",
                    "tmin",
                    "prob_lluvia",
                    "clientes_estimados",
                    "intervalo_inferior",
                    "intervalo_superior",
                    "empleados_recomendados",
                ]
            ].rename(
                columns={
                    "fecha": "Fecha",
                    "dia": "Día",
                    "abierto": "Abierto",
                    "motivo_cierre": "Motivo del cierre",
                    "tmax": "T. máx (°C)",
                    "tmin": "T. mín (°C)",
                    "prob_lluvia": "Lluvia (%)",
                    "clientes_estimados": "Clientes",
                    "intervalo_inferior": "Mínimo probable",
                    "intervalo_superior": "Máximo probable",
                    "empleados_recomendados": "Empleados",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "El intervalo cubre 8 de cada 10 días y se ensancha "
            "con la previsión: un sábado lleno es más incierto "
            "que un miércoles flojo."
        )


# ============================================================
# FIABILIDAD DEL MODELO
# ============================================================

st.divider()

st.subheader("Hasta dónde fiarse de esta cifra")

metricas = metadatos.get("metricas_test", {})

columnas = st.columns(4)

columnas[0].metric(
    "Error medio",
    f"{metricas.get('mae', float('nan')):.1f} clientes",
    help=(
        "Diferencia media entre lo previsto y lo que "
        "realmente ocurrió, medida sobre días que el modelo "
        "no había visto nunca."
    ),
)

columnas[1].metric(
    "Error relativo",
    f"{metricas.get('mape', float('nan')):.1f} %",
)

columnas[2].metric(
    "Variabilidad explicada",
    f"{metricas.get('r2', float('nan')):.0%}",
    help=(
        "Proporción de la variación diaria de clientes que el "
        "modelo consigue explicar."
    ),
)

columnas[3].metric(
    "Modelo",
    metadatos.get("algoritmo", "-")
    .replace("_", " ")
    .capitalize(),
    help=(
        "Entrenado con "
        f"{metadatos.get('registros_entrenamiento', '?')} días "
        "de actividad real."
    ),
)

sesgo = metadatos.get("sesgo_test")

if sesgo and sesgo > 3:

    st.warning(
        f"**Limitación conocida.** El modelo se queda corto "
        f"una media de {sesgo:.0f} clientes, porque el negocio "
        "crece más deprisa de lo que sabe extrapolar. Si la "
        "decisión es ajustada, tira hacia el extremo alto del "
        "rango. El sesgo desaparece reentrenando el modelo "
        "cada pocos meses."
    )


# ============================================================
# DETALLE TÉCNICO
# ============================================================

with st.expander("Detalle técnico del modelo"):

    st.markdown(
        f"""
        **Algoritmo:** {metadatos.get('algoritmo', '-')}

        **Entrenado con:**
        {metadatos.get('registros_entrenamiento', '?')} días
        ({metadatos['periodo_entrenamiento'][0]} a
        {metadatos['periodo_entrenamiento'][1]})

        **Evaluado con:** días de 2026 que el modelo nunca vio
        durante el entrenamiento.

        **Intervalo:** calculado sobre los errores reales del
        último año completo, sin suponer que sigan una
        distribución normal.
        """
    )

    if comparacion is not None:

        st.markdown("**Comparación de modelos**")

        st.dataframe(
            comparacion[
                [
                    "modelo",
                    "mae_validacion",
                    "mae_test",
                    "r2_test",
                ]
            ].rename(
                columns={
                    "modelo": "Modelo",
                    "mae_validacion": "Error (validación)",
                    "mae_test": "Error (test)",
                    "r2_test": "R² (test)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    grafico = ruta_grafico("serie_real_vs_predicho")

    if grafico.exists():

        st.markdown("**Previsión frente a realidad en 2026**")

        st.image(str(grafico), use_container_width=True)


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Trabajo Fin de Máster · Predicción de demanda en "
    "hostelería a partir de meteorología y calendario  \n"
    "Juan Francisco Morales Olivo · Datos meteorológicos: "
    "AEMET OpenData (estación 3182Y, Arganda del Rey)"
)
