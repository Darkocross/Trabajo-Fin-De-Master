# src/models/prediccion_manana.py

# ============================================================
# PREDICCIÓN PARA LOS PRÓXIMOS DÍAS
# ============================================================
#
# Genera la predicción de demanda usando la PREVISIÓN
# meteorológica de AEMET, no la meteorología observada.
#
# Esta distinción es la que hace que el sistema sea utilizable
# de verdad. El modelo se entrena con la temperatura y la
# lluvia que REALMENTE hizo cada día, porque es lo único que
# existe del pasado. Pero cuando el encargado tiene que decidir
# a quién llama para el sábado, todavía no sabe qué tiempo
# hará: solo tiene la previsión.
#
# Por eso la predicción se alimenta de la previsión a 7 días
# del municipio, con dos limitaciones que conviene tener
# presentes:
#
#   1. La previsión tiene su propio error, que se suma al del
#      modelo. Cuanto más lejos el día, mayor es.
#
#   2. AEMET no publica los milímetros previstos en la
#      predicción municipal: solo la PROBABILIDAD de
#      precipitación. Hay que estimar los milímetros a partir
#      de esa probabilidad (ver `estimar_precipitacion`).
#
# ============================================================

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    GOLD_DATASET,
    PREDICCION_AEMET,
    PREDICCION_MANANA,
    PROCESSED_RESULTADOS_DIR,
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
# LECTURA DE LA PREVISIÓN DE AEMET
# ============================================================

def cargar_prediccion_aemet():
    """
    Carga el JSON de previsión descargado de AEMET.
    """

    if not PREDICCION_AEMET.exists():

        raise FileNotFoundError(
            f"No existe la previsión de AEMET:\n"
            f"{PREDICCION_AEMET}\n\n"
            "Descárgala con:\n"
            "    python -m src.extraction.descargar_prediccion"
        )

    with open(
        PREDICCION_AEMET,
        "r",
        encoding="utf-8",
    ) as archivo:

        datos = json.load(archivo)

    if isinstance(datos, list) and datos:
        datos = datos[0]

    dias = (
        datos
        .get("prediccion", {})
        .get("dia", [])
    )

    if not dias:

        raise ValueError(
            "La previsión de AEMET no contiene días."
        )

    return datos, dias


def valor_periodo_completo(lista, clave="value"):
    """
    De las listas que devuelve AEMET por tramos horarios,
    toma la del periodo completo "00-24".
    """

    if not isinstance(lista, list):
        return None

    for elemento in lista:

        if elemento.get("periodo") in ("00-24", None, ""):

            valor = elemento.get(clave)

            if valor not in (None, ""):
                return valor

    for elemento in lista:

        valor = elemento.get(clave)

        if valor not in (None, ""):
            return valor

    return None


# ============================================================
# ESTIMACIÓN DE LA PRECIPITACIÓN
# ============================================================

def calcular_lluvia_media_dia_lluvioso():
    """
    Calcula, sobre el histórico observado, cuánto llueve de
    media en los días en los que efectivamente llueve.

    Se usa para convertir la probabilidad de precipitación en
    una estimación de milímetros.
    """

    if not GOLD_DATASET.exists():
        return 5.0

    gold = pd.read_csv(GOLD_DATASET)

    lluviosos = gold[gold["prec"] > 0.5]

    if lluviosos.empty:
        return 5.0

    return float(lluviosos["prec"].mean())


def estimar_precipitacion(
    probabilidad,
    lluvia_media_dia_lluvioso,
):
    """
    Convierte la probabilidad de precipitación de AEMET en una
    estimación de milímetros.

    AEMET publica en la predicción municipal la probabilidad
    de que llueva, pero no cuánto. Se usa el valor esperado:

        mm estimados = P(llueva) x mm medios de un día lluvioso

    Es una aproximación, no una medición, y así se documenta.
    Para el uso que le damos es suficiente: al modelo le
    importa sobre todo distinguir un día seco de uno lluvioso,
    no acertar el milímetro exacto.
    """

    if probabilidad is None:
        return 0.0

    try:
        probabilidad = float(probabilidad)

    except (TypeError, ValueError):
        return 0.0

    probabilidad = max(0.0, min(100.0, probabilidad)) / 100

    return round(
        probabilidad * lluvia_media_dia_lluvioso,
        1,
    )


# ============================================================
# EXTRACCIÓN DE LA METEOROLOGÍA PREVISTA
# ============================================================

def extraer_meteorologia(dia, lluvia_media_dia_lluvioso):
    """
    Extrae de un día de la previsión de AEMET las variables
    que necesita el modelo.
    """

    fecha = pd.Timestamp(dia["fecha"]).normalize()

    temperatura = dia.get("temperatura", {})

    tmax = temperatura.get("maxima")
    tmin = temperatura.get("minima")

    if tmax is None or tmin is None:
        return None

    probabilidad = valor_periodo_completo(
        dia.get("probPrecipitacion", [])
    )

    prec = estimar_precipitacion(
        probabilidad,
        lluvia_media_dia_lluvioso,
    )

    estado_cielo = valor_periodo_completo(
        dia.get("estadoCielo", []),
        clave="descripcion",
    )

    return {
        "fecha": fecha,
        "tmax": float(tmax),
        "tmin": float(tmin),
        "prec": prec,
        "probabilidad_lluvia": (
            float(probabilidad)
            if probabilidad is not None
            else 0.0
        ),
        "estado_cielo": estado_cielo or "",
    }


# ============================================================
# MAIN
# ============================================================

# ============================================================
# PREVISIÓN DE LOS PRÓXIMOS DÍAS
# ============================================================

def prevision_proximos_dias(
    dias_a_predecir=7,
    modelo=None,
    metadatos=None,
    calendario=None,
    gold=None,
):
    """
    Devuelve la previsión de clientes de los próximos días a
    partir de la predicción meteorológica descargada de AEMET.

    Esta función NO imprime nada ni escribe ficheros: solo
    calcula. Así la puede usar tanto el script de consola como
    la aplicación, que necesita el resultado en memoria y no un
    CSV que puede estar desactualizado.

    Los recursos (modelo, calendario, histórico) se pueden
    pasar ya cargados. La aplicación los tiene cacheados y
    volver a leerlos en cada interacción haría que la pantalla
    tardara un segundo en responder a cada clic.

    Devuelve (DataFrame, info), donde info trae el municipio,
    la fecha de elaboración y si la previsión está caducada.
    """

    datos, dias = cargar_prediccion_aemet()

    if modelo is None or metadatos is None:
        modelo, metadatos = cargar_modelo_produccion()

    if calendario is None:
        calendario = cargar_calendario()

    if gold is None and GOLD_DATASET.exists():
        gold = pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])

    lluvia_media = calcular_lluvia_media_dia_lluvioso()

    hoy = pd.Timestamp(date.today()).normalize()

    meteorologias = [
        meteo
        for meteo in (
            extraer_meteorologia(dia, lluvia_media)
            for dia in dias[:dias_a_predecir]
        )
        if meteo is not None
    ]

    futuros = [
        meteo
        for meteo in meteorologias
        if meteo["fecha"] >= hoy
    ]

    # AEMET publica solo 7 días. Si el fichero descargado ya ha
    # quedado atrás se usan igualmente sus días, para poder
    # revisar el sistema sin depender de la conexión, pero se
    # marca como caducada.

    seleccionados = futuros if futuros else meteorologias

    caducada = not futuros

    filas = []

    for meteo in seleccionados:

        resultado = predecir(
            fecha=meteo["fecha"],
            tmax=meteo["tmax"],
            tmin=meteo["tmin"],
            prec=meteo["prec"],
            modelo=modelo,
            metadatos=metadatos,
            calendario=calendario,
        )

        historico = contexto_historico(meteo["fecha"], gold)

        cerrado = resultado["cerrado"]

        filas.append({
            "fecha": meteo["fecha"].date(),
            "dia": resultado["nombre_dia"],
            "abierto": 0 if cerrado else 1,
            "motivo_cierre": (
                resultado["motivo_cierre"] if cerrado else ""
            ),
            "tmax": meteo["tmax"],
            "tmin": meteo["tmin"],
            "prob_lluvia": meteo["probabilidad_lluvia"],
            "prec_estimada": meteo["prec"],
            "estado_cielo": meteo.get("estado_cielo", ""),
            "clientes_estimados": (
                None if cerrado
                else round(resultado["prediccion"], 1)
            ),
            "intervalo_inferior": (
                None if cerrado
                else round(resultado["intervalo_inferior"], 1)
            ),
            "intervalo_superior": (
                None if cerrado
                else round(resultado["intervalo_superior"], 1)
            ),
            "empleados_recomendados": (
                0 if cerrado
                else resultado["empleados_recomendados"]
            ),
            "media_historica": (
                round(historico["media_dia_semana"], 1)
                if historico else None
            ),
        })

    if not filas:
        return None, None

    resultados = pd.DataFrame(filas)

    resultados["prevision_caducada"] = int(caducada)

    info = {
        "municipio": datos.get("nombre", "-"),
        "provincia": datos.get("provincia", "-"),
        "elaborado": datos.get("elaborado", "-"),
        "caducada": caducada,
    }

    return resultados, info


def main(dias_a_predecir=7):

    print("=" * 60)
    print("PREDICCIÓN DE DEMANDA")
    print("=" * 60)

    # --------------------------------------------------------
    # CARGAR RECURSOS
    # --------------------------------------------------------

    datos, dias = cargar_prediccion_aemet()

    print()
    print(f"Municipio: {datos.get('nombre', '-')}")
    print(f"Provincia: {datos.get('provincia', '-')}")
    print(f"Elaborada: {datos.get('elaborado', '-')}")

    modelo, metadatos = cargar_modelo_produccion()

    print()
    print(
        f"Modelo: {metadatos.get('algoritmo', 'desconocido')}"
    )

    print(
        "Entrenado con "
        f"{metadatos.get('registros_entrenamiento', '?')} días "
        f"({metadatos['periodo_entrenamiento'][0]} -> "
        f"{metadatos['periodo_entrenamiento'][1]})"
    )

    calendario = cargar_calendario()

    gold = (
        pd.read_csv(GOLD_DATASET, parse_dates=["fecha"])
        if GOLD_DATASET.exists()
        else None
    )

    lluvia_media = calcular_lluvia_media_dia_lluvioso()

    print()
    print(
        "Lluvia media de un día lluvioso en el histórico: "
        f"{lluvia_media:.1f} mm"
    )

    print(
        "(se usa para estimar los mm a partir de la "
        "probabilidad de AEMET)"
    )

    # --------------------------------------------------------
    # PREDECIR CADA DÍA
    # --------------------------------------------------------

    hoy = pd.Timestamp(date.today()).normalize()

    meteorologias = [
        meteo
        for meteo in (
            extraer_meteorologia(dia, lluvia_media)
            for dia in dias[:dias_a_predecir]
        )
        if meteo is not None
    ]

    futuros = [
        meteo
        for meteo in meteorologias
        if meteo["fecha"] >= hoy
    ]

    # --------------------------------------------------------
    # PREVISIÓN CADUCADA
    # --------------------------------------------------------
    #
    # AEMET publica solo 7 días. Si el fichero descargado ya
    # ha quedado atrás, se avisa y se predicen igualmente los
    # días que contiene, para poder revisar el funcionamiento
    # del sistema sin depender de la conexión.

    if futuros:

        seleccionados = futuros
        caducada = False

    else:

        seleccionados = meteorologias
        caducada = True

        print()
        print("AVISO")
        print("-" * 60)

        print(
            "La previsión descargada es anterior a hoy "
            f"({hoy.date()}). Se muestran igualmente sus días "
            "para poder comprobar el sistema."
        )

        print()
        print("Para una predicción real, descarga la actual:")
        print(
            "    python -m src.extraction.descargar_prediccion"
        )

    filas = []

    for meteo in seleccionados:

        resultado = predecir(
            fecha=meteo["fecha"],
            tmax=meteo["tmax"],
            tmin=meteo["tmin"],
            prec=meteo["prec"],
            modelo=modelo,
            metadatos=metadatos,
            calendario=calendario,
        )

        historico = contexto_historico(
            meteo["fecha"],
            gold,
        )

        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        print()
        print("=" * 60)

        print(
            f"{meteo['fecha'].date()}  ·  "
            f"{resultado['nombre_dia']}"
        )

        print("=" * 60)

        if resultado["cerrado"]:

            print()
            print(
                "El restaurante está CERRADO "
                f"({resultado['motivo_cierre']})."
            )

            filas.append({
                "fecha": meteo["fecha"].date(),
                "dia": resultado["nombre_dia"],
                "abierto": 0,
                "motivo_cierre": resultado["motivo_cierre"],
                "tmax": meteo["tmax"],
                "tmin": meteo["tmin"],
                "prob_lluvia": meteo["probabilidad_lluvia"],
                "prec_estimada": meteo["prec"],
                "clientes_estimados": None,
                "intervalo_inferior": None,
                "intervalo_superior": None,
                "empleados_recomendados": 0,
            })

            continue

        print()
        print(
            f"Previsión: {meteo['tmin']:.0f} - "
            f"{meteo['tmax']:.0f} °C   ·   "
            f"{meteo['probabilidad_lluvia']:.0f} % de lluvia "
            f"(~{meteo['prec']:.1f} mm)   ·   "
            f"{meteo['estado_cielo']}"
        )

        print()
        print(
            f"CLIENTES ESTIMADOS: "
            f"{resultado['prediccion']:.0f}"
        )

        print(
            f"Intervalo (80 %):   "
            f"{resultado['intervalo_inferior']:.0f} - "
            f"{resultado['intervalo_superior']:.0f}"
        )

        empleados = resultado["empleados_recomendados"]

        estado, titulo, detalle = evaluar_plantilla(
            resultado["prediccion"],
            empleados,
        )

        print()
        print(
            f"PLANTILLA RECOMENDADA: {empleados} empleados "
            f"({detalle})"
        )

        if historico:

            print()
            print(
                "Referencia histórica de este día: "
                f"{historico['media_dia_semana']:.0f} clientes "
                "de media"
            )

        print()
        print("Por qué:")

        for motivo in explicar(resultado, historico):
            print(f"  - {motivo}")

        filas.append({
            "fecha": meteo["fecha"].date(),
            "dia": resultado["nombre_dia"],
            "abierto": 1,
            "motivo_cierre": "",
            "tmax": meteo["tmax"],
            "tmin": meteo["tmin"],
            "prob_lluvia": meteo["probabilidad_lluvia"],
            "prec_estimada": meteo["prec"],
            "clientes_estimados": round(
                resultado["prediccion"], 1
            ),
            "intervalo_inferior": round(
                resultado["intervalo_inferior"], 1
            ),
            "intervalo_superior": round(
                resultado["intervalo_superior"], 1
            ),
            "empleados_recomendados": empleados,
        })

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    if not filas:

        print()
        print(
            "La previsión descargada no contiene ningún día "
            "utilizable. Vuelve a descargarla con:"
        )
        print(
            "    python -m src.extraction.descargar_prediccion"
        )

        return None

    resultados = pd.DataFrame(filas)

    resultados["prevision_caducada"] = int(caducada)

    PROCESSED_RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados.to_csv(
        PREDICCION_MANANA,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("RESUMEN DE LA SEMANA")
    print("=" * 60)
    print()

    print(resultados.to_string(index=False))

    print()
    print(f"Guardado en: {PREDICCION_MANANA}")

    return resultados


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
