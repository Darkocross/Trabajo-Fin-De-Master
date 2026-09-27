# tests/test_gold.py

# ============================================================
# TESTS DE LA CAPA GOLD
# ============================================================
#
# La capa gold es el contrato entre la parte de datos y la
# parte de modelado. Estos tests comprueban que ese contrato
# se cumple: que están todas las columnas, que los valores son
# coherentes y que no se ha colado ningún día que no debería
# estar.
#
# ============================================================

import pandas as pd
import pytest

from src.config import ACTIVIDAD_HOSTELERIA, MES_CIERRE_VACACIONES
from src.transformation.crear_gold import (
    COLUMNAS_GOLD,
    COLUMNAS_HOSTELERIA,
    completar_meteorologia,
    obtener_tipo_vacaciones,
)


# ============================================================
# SEPARACIÓN DE FUENTES EN LA CAPA RAW
# ============================================================
#
# Cada fuente de la capa raw debe contener únicamente lo suyo.
# Si la actividad del restaurante incluyera la meteorología,
# habría dos copias del mismo dato que podrían divergir, y
# dejaría de estar claro de dónde sale cada columna.

def test_la_actividad_no_contiene_meteorologia():
    """
    Un TPV no mide la temperatura. La meteorología tiene su
    propia fuente en raw/meteorologia/.
    """

    if not ACTIVIDAD_HOSTELERIA.exists():

        pytest.skip(
            "No existen los datos de actividad. Ejecuta: "
            "python pipeline.py --hasta hosteleria"
        )

    actividad = pd.read_csv(ACTIVIDAD_HOSTELERIA, nrows=1)

    meteorologicas = ["tmed", "tmax", "tmin", "prec"]

    intrusas = [
        columna
        for columna in meteorologicas
        if columna in actividad.columns
    ]

    assert not intrusas, (
        f"La capa de actividad contiene {intrusas}, que "
        "pertenecen a raw/meteorologia/"
    )


def test_la_actividad_no_contiene_calendario():
    """
    El día de la semana y el periodo vacacional son
    calendario, no actividad. Se derivan de la fecha o se
    toman del calendario al construir la capa gold.
    """

    if not ACTIVIDAD_HOSTELERIA.exists():
        pytest.skip("Ejecuta antes: python pipeline.py")

    actividad = pd.read_csv(ACTIVIDAD_HOSTELERIA, nrows=1)

    calendario = [
        "dia_semana",
        "tipo_vacaciones",
        "es_festivo",
        "es_vacaciones",
    ]

    intrusas = [
        columna
        for columna in calendario
        if columna in actividad.columns
    ]

    assert not intrusas, (
        f"La capa de actividad contiene {intrusas}, que "
        "pertenecen al calendario"
    )


def test_la_actividad_tiene_exactamente_lo_que_registra_el_local():
    """
    Fecha, plantilla, clientes y nota de la jornada. Nada más.
    """

    if not ACTIVIDAD_HOSTELERIA.exists():
        pytest.skip("Ejecuta antes: python pipeline.py")

    actividad = pd.read_csv(ACTIVIDAD_HOSTELERIA, nrows=1)

    assert list(actividad.columns) == COLUMNAS_HOSTELERIA


# ============================================================
# TIPO DE VACACIONES
# ============================================================

@pytest.mark.parametrize(
    "periodo,esperado",
    [
        ("Vacaciones de Navidad", "navidad"),
        ("Vacaciones de Semana Santa", "semana_santa"),
        ("Vacaciones de verano", "verano"),
        ("Puente de mayo", "otros"),
        (None, "ninguna"),
        ("", "ninguna"),
        ("nan", "ninguna"),
        (float("nan"), "ninguna"),
    ],
)
def test_tipo_vacaciones(periodo, esperado):

    assert obtener_tipo_vacaciones(periodo) == esperado


# ============================================================
# TRATAMIENTO DE HUECOS METEOROLÓGICOS
# ============================================================

def test_completar_meteorologia_interpola_temperaturas():
    """
    Un hueco de un día entre 10 y 20 grados debe interpolarse
    a 15, no descartarse.
    """

    datos = pd.DataFrame({
        "fecha": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"]
        ),
        "tmed": [10.0, None, 20.0],
        "tmax": [15.0, None, 25.0],
        "tmin": [5.0, None, 15.0],
        "prec": [0.0, 0.0, 0.0],
    })

    completado = completar_meteorologia(datos)

    assert completado["tmed"].iloc[1] == pytest.approx(15.0)
    assert completado["tmax"].iloc[1] == pytest.approx(20.0)
    assert completado["tmin"].iloc[1] == pytest.approx(10.0)

    assert completado["tmed"].notna().all()


def test_completar_meteorologia_rellena_lluvia_con_cero():
    """
    AEMET marca como "Ip" la lluvia inapreciable, que al pasar
    a número queda vacía. Debe tratarse como 0 mm.
    """

    datos = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "tmed": [10.0, 12.0],
        "tmax": [15.0, 17.0],
        "tmin": [5.0, 7.0],
        "prec": [None, 3.0],
    })

    completado = completar_meteorologia(datos)

    assert completado["prec"].iloc[0] == 0.0


def test_completar_meteorologia_marca_los_imputados():

    datos = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "tmed": [10.0, None],
        "tmax": [15.0, 17.0],
        "tmin": [5.0, 7.0],
        "prec": [0.0, 0.0],
    })

    completado = completar_meteorologia(datos)

    assert completado["meteo_imputada"].tolist() == [0, 1]


# ============================================================
# ESTRUCTURA DE LA CAPA GOLD
# ============================================================

def test_columnas_esperadas(gold):

    assert list(gold.columns) == COLUMNAS_GOLD


def test_una_fila_por_dia(gold):

    assert gold["fecha"].is_unique

    assert gold["fecha"].is_monotonic_increasing


def test_sin_nulos_en_columnas_criticas(gold):

    criticas = [
        "fecha",
        "dia_semana",
        "mes",
        "n_clientes",
        "tmed",
        "tmax",
        "tmin",
        "prec",
        "tipo_vacaciones",
    ]

    assert gold[criticas].isna().sum().sum() == 0


# ============================================================
# COHERENCIA
# ============================================================

def test_dia_semana_y_mes_coinciden_con_la_fecha(gold):

    assert (
        gold["dia_semana"] == gold["fecha"].dt.weekday
    ).all()

    assert (gold["mes"] == gold["fecha"].dt.month).all()

    assert (gold["año"] == gold["fecha"].dt.year).all()


def test_temperaturas_ordenadas(gold):
    """
    La mínima nunca puede superar a la media, ni la media a la
    máxima.
    """

    assert (gold["tmin"] <= gold["tmed"]).all()

    assert (gold["tmed"] <= gold["tmax"]).all()


def test_tmed_es_el_punto_medio(gold):
    """
    AEMET calcula la temperatura media como el punto medio
    entre máxima y mínima.

    Es el hecho en el que se apoya la decisión de sustituir
    las tres temperaturas por `tmed` y `amplitud_termica`. Si
    algún día dejara de cumplirse, esa decisión habría que
    revisarla.
    """

    punto_medio = (gold["tmax"] + gold["tmin"]) / 2

    assert (gold["tmed"] - punto_medio).abs().max() < 0.2


def test_sin_dias_de_cierre(gold):
    """
    La capa gold solo contiene días de actividad.
    """

    assert (gold["mes"] != MES_CIERRE_VACACIONES).all()

    assert (gold["n_clientes"] > 0).all()


def test_valores_en_rangos_razonables(gold):

    assert (gold["prec"] >= 0).all()

    assert gold["n_empleados"].between(1, 20).all()

    assert gold["tmin"].between(-20, 50).all()

    assert gold["tmax"].between(-20, 55).all()


def test_tipo_vacaciones_solo_admite_categorias_conocidas(gold):

    categorias = {
        "ninguna",
        "navidad",
        "semana_santa",
        "verano",
        "otros",
    }

    assert set(gold["tipo_vacaciones"].unique()) <= categorias


def test_no_hay_fechas_futuras(gold):

    assert gold["fecha"].max() <= pd.Timestamp.today()


# ============================================================
# PATRONES DE NEGOCIO
# ============================================================

def test_los_fines_de_semana_tienen_mas_clientes(gold):
    """
    Comprobación de sentido común: si el sábado no tuviese más
    gente que el martes, algo estaría mal en los datos.
    """

    fin_de_semana = gold[gold["fin_de_semana"] == 1]

    entre_semana = gold[gold["fin_de_semana"] == 0]

    assert (
        fin_de_semana["n_clientes"].mean()
        > entre_semana["n_clientes"].mean()
    )


def test_llover_reduce_la_demanda(gold):
    """
    La hipótesis central del proyecto debe verse en los datos.
    """

    lluviosos = gold[gold["prec"] > 5]

    secos = gold[gold["prec"] == 0]

    assert len(lluviosos) > 20, "muy pocos días de lluvia"

    assert (
        lluviosos["n_clientes"].mean()
        < secos["n_clientes"].mean()
    )


# ============================================================
# INTERPRETACIÓN DE LOS CÓDIGOS DE AEMET
# ============================================================
#
# AEMET no solo devuelve números. La capa raw los conserva tal
# cual y es aquí donde se deciden. Estos tests fijan esa
# decisión, que es de análisis y no de formato.

def test_ip_es_una_medicion_no_un_hueco():
    """
    "Ip" significa que llovió menos de 0,1 mm. Es un dato de
    AEMET, así que vale 0,0 mm y NO cuenta como imputación.
    """

    import pandas as pd

    from src.transformation.crear_gold import (
        completar_meteorologia,
    )

    meteorologia = pd.DataFrame({
        "fecha": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]),
        "tmed": ["10.0", "11.0", "12.0"],
        "tmax": ["15.0", "16.0", "17.0"],
        "tmin": ["5.0", "6.0", "7.0"],
        "prec": ["0.0", "Ip", "2.5"],
    })

    resultado = completar_meteorologia(meteorologia)

    fila = resultado.iloc[1]

    assert fila["prec"] == 0.0
    assert fila["prec_inapreciable"] == 1

    # Lo importante: el dato es de AEMET, no nuestro.
    assert fila["prec_rellenada"] == 0
    assert fila["meteo_imputada"] == 0


def test_el_hueco_de_verdad_si_se_marca():
    """
    Una celda vacía sí es un hueco: se rellena con 0,0 mm y se
    marca, para poder excluir esos días del análisis.
    """

    import pandas as pd

    from src.transformation.crear_gold import (
        completar_meteorologia,
    )

    meteorologia = pd.DataFrame({
        "fecha": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ]),
        "tmed": ["10.0", None, "12.0"],
        "tmax": ["15.0", "16.0", "17.0"],
        "tmin": ["5.0", "6.0", "7.0"],
        "prec": ["0.0", None, "2.5"],
    })

    resultado = completar_meteorologia(meteorologia)

    fila = resultado.iloc[1]

    assert fila["prec"] == 0.0
    assert fila["prec_rellenada"] == 1
    assert fila["temp_interpolada"] == 1
    assert fila["meteo_imputada"] == 1

    # La temperatura interpolada cae entre sus vecinas.
    assert 10.0 <= fila["tmed"] <= 12.0

    # Y los días completos no se marcan.
    assert resultado.iloc[0]["meteo_imputada"] == 0
    assert resultado.iloc[2]["meteo_imputada"] == 0


def test_los_valores_normales_no_se_tocan():
    """
    Regresión de un error real: al interpretar "Ip" con
    `mask`, los valores nulos se trataban como coincidencia y
    TODA la columna se convertía en ceros, en silencio.
    """

    import pandas as pd

    from src.transformation.crear_gold import (
        interpretar_medida,
    )

    serie = pd.Series(["0.0", "12.4", None, "Ip", "3.2"])

    numerico, marca = interpretar_medida(serie)

    assert numerico.iloc[1] == 12.4
    assert numerico.iloc[4] == 3.2

    # El nulo sigue siendo nulo, no un cero.
    assert pd.isna(numerico.iloc[2])
    assert not marca.iloc[2]

    # Y solo "Ip" está marcado.
    assert marca.tolist() == [
        False, False, False, True, False
    ]

