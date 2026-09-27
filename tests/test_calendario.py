# tests/test_calendario.py

# ============================================================
# TESTS DEL CALENDARIO
# ============================================================
#
# El calendario es la pieza de la que dependen todas las demás:
# define qué días abre el restaurante, cuáles son festivos y
# qué variables de calendario recibe el modelo. Si se rompe, se
# rompe el proyecto entero sin dar ningún error visible.
#
# ============================================================

from datetime import date

import pytest

from src.transformation.generar_calendario import (
    domingo_de_pascua,
    festivos_del_año,
    generar_calendario,
    jueves_santo,
    trasladar_si_domingo,
    viernes_santo,
)


# ============================================================
# CÁLCULO DE LA PASCUA
# ============================================================

# Fechas oficiales del Domingo de Resurrección.
PASCUAS = {
    2022: date(2022, 4, 17),
    2023: date(2023, 4, 9),
    2024: date(2024, 3, 31),
    2025: date(2025, 4, 20),
    2026: date(2026, 4, 5),
    2027: date(2027, 3, 28),
}


@pytest.mark.parametrize("año,esperada", PASCUAS.items())
def test_domingo_de_pascua(año, esperada):
    """
    El algoritmo de Butcher-Meeus debe dar las fechas
    oficiales.
    """

    assert domingo_de_pascua(año) == esperada


def test_pascua_siempre_en_domingo():
    """
    Comprobación de coherencia sobre un siglo entero.
    """

    for año in range(2000, 2101):

        assert domingo_de_pascua(año).weekday() == 6


def test_jueves_y_viernes_santo():
    """
    Jueves y Viernes Santo son los días anteriores a la Pascua.
    """

    for año in PASCUAS:

        assert jueves_santo(año).weekday() == 3
        assert viernes_santo(año).weekday() == 4

        assert (
            domingo_de_pascua(año) - viernes_santo(año)
        ).days == 2


# ============================================================
# TRASLADO DE FESTIVOS
# ============================================================

def test_festivo_en_domingo_se_traslada_al_lunes():

    domingo = date(2026, 11, 1)

    assert domingo.weekday() == 6

    assert trasladar_si_domingo(domingo) == date(2026, 11, 2)


def test_festivo_entre_semana_no_se_mueve():

    lunes = date(2026, 10, 12)

    assert trasladar_si_domingo(lunes) == lunes


# ============================================================
# FESTIVOS
# ============================================================

def test_numero_de_festivos_por_año():
    """
    El calendario laboral español tiene 14 festivos: 12 entre
    nacionales y autonómicos, más 2 locales.
    """

    for año in range(2022, 2027):

        assert len(festivos_del_año(año)) == 14


def test_festivos_oficiales_de_2026():
    """
    Contraste con el calendario laboral publicado para
    Arganda del Rey en 2026.
    """

    festivos = festivos_del_año(2026)

    esperados = [
        date(2026, 1, 1),    # Año Nuevo
        date(2026, 1, 6),    # Epifanía
        date(2026, 4, 2),    # Jueves Santo
        date(2026, 4, 3),    # Viernes Santo
        date(2026, 5, 1),    # Día del Trabajo
        date(2026, 5, 2),    # Comunidad de Madrid
        date(2026, 8, 15),   # Asunción
        date(2026, 9, 8),    # Virgen de la Soledad (local)
        date(2026, 9, 9),    # Fiesta local
        date(2026, 10, 12),  # Fiesta Nacional
        date(2026, 11, 2),   # Todos los Santos (trasladado)
        date(2026, 12, 7),   # Constitución (trasladado)
        date(2026, 12, 8),   # Inmaculada
        date(2026, 12, 25),  # Navidad
    ]

    assert sorted(festivos) == esperados


def test_fiestas_locales_de_arganda():
    """
    Las fiestas patronales de Arganda del Rey son en
    septiembre.
    """

    for año in range(2022, 2027):

        festivos = festivos_del_año(año)

        locales = [
            fecha
            for fecha, (nombre, ambito) in festivos.items()
            if ambito == "local"
        ]

        assert len(locales) == 2

        assert all(fecha.month == 9 for fecha in locales)


# ============================================================
# GENERACIÓN COMPLETA
# ============================================================

@pytest.fixture(scope="module")
def calendario_generado():

    return generar_calendario(2022, 2026)


def test_calendario_cubre_todos_los_dias(calendario_generado):
    """
    2022-2026 son 1826 días (2024 es bisiesto).
    """

    assert len(calendario_generado) == 1826

    assert calendario_generado["fecha"].is_unique


def test_dia_semana_coincide_con_la_fecha(calendario_generado):

    import pandas as pd

    fechas = pd.to_datetime(calendario_generado["fecha"])

    assert (
        calendario_generado["dia_semana_num"]
        == fechas.dt.weekday
    ).all()


def test_fin_de_semana_es_sabado_o_domingo(
    calendario_generado,
):

    fines = calendario_generado[
        calendario_generado["fin_de_semana"] == 1
    ]

    assert fines["dia_semana_num"].isin([5, 6]).all()


def test_agosto_esta_cerrado(calendario_generado):
    """
    El restaurante cierra por vacaciones todo agosto.
    """

    import pandas as pd

    fechas = pd.to_datetime(calendario_generado["fecha"])

    agosto = calendario_generado[fechas.dt.month == 8]

    assert (agosto["restaurante_cerrado"] == 1).all()


def test_no_se_cierra_por_descanso_en_festivo(
    calendario_generado,
):
    """
    Regla de negocio: el descanso semanal de lunes y martes no
    se aplica cuando ese día es festivo.
    """

    festivos = calendario_generado[
        calendario_generado["es_festivo"] == 1
    ]

    assert (festivos["cerrado_descanso"] == 0).all()


def test_el_calendario_no_publica_hipotesis(
    calendario_generado,
):
    """
    El calendario contiene hechos verificables contra el
    almanaque, no hipótesis de modelado.

    `es_mayo`, `es_junio`, `es_julio` y sus interacciones con
    el fin de semana venían de los meses que amplifica el
    generador sintético, no de los datos. Las interacciones
    viven ahora en src/models/features.py, donde se pueden
    medir.
    """

    hipotesis = [
        "es_semana_santa",
        "es_mayo",
        "es_junio",
        "es_julio",
        "es_fin_semana_semana_santa",
        "es_fin_semana_mayo",
        "es_fin_semana_junio",
        "es_fin_semana_julio",
    ]

    presentes = [
        columna
        for columna in hipotesis
        if columna in calendario_generado.columns
    ]

    assert not presentes, (
        "El calendario ha vuelto a publicar hipótesis de "
        f"modelado: {presentes}"
    )


def test_ambito_de_los_festivos(calendario_generado):
    """
    Todo festivo lleva su ámbito, y solo los festivos.
    """

    calendario = calendario_generado

    festivos = calendario[calendario["es_festivo"] == 1]
    laborables = calendario[calendario["es_festivo"] == 0]

    assert festivos["ambito_festivo"].notna().all()

    assert laborables["ambito_festivo"].isna().all()

    ambitos = set(festivos["ambito_festivo"].unique())

    assert ambitos <= {"nacional", "autonomico", "local"}


def test_visperas_y_puentes(calendario_generado):
    """
    Víspera y puente se comprueban contra el propio almanaque.

    - Una víspera es el día anterior a un festivo.
    - Un puente es un día laborable (ni festivo ni fin de
      semana) encajonado entre un festivo y el fin de semana.
    """

    import pandas as pd

    calendario = calendario_generado.copy()

    calendario["fecha"] = pd.to_datetime(calendario["fecha"])

    calendario = calendario.sort_values("fecha")

    festivo_siguiente = (
        calendario["es_festivo"].shift(-1).fillna(0)
    )

    # El último día del periodo no tiene día siguiente dentro
    # del calendario, así que se excluye de la comprobación.

    comparables = calendario.iloc[:-1]

    assert (
        comparables["es_vispera_festivo"]
        == festivo_siguiente.iloc[:-1].astype(int)
    ).all()

    puentes = calendario[calendario["es_puente"] == 1]

    assert (puentes["es_festivo"] == 0).all()

    assert (puentes["fin_de_semana"] == 0).all()

    # Solo lunes y viernes pueden ser puente.
    assert puentes["dia_semana_num"].isin([0, 4]).all()


def test_semana_santa_contiene_la_pascua(calendario_generado):
    """
    El periodo vacacional de Semana Santa debe incluir el
    Domingo de Resurrección de cada año.
    """

    import pandas as pd

    fechas = pd.to_datetime(calendario_generado["fecha"])

    for año in range(2022, 2027):

        pascua = pd.Timestamp(domingo_de_pascua(año))

        fila = calendario_generado[fechas == pascua]

        assert not fila.empty

        periodo = fila["periodo_vacacional"].iloc[0]

        assert "Semana Santa" in str(periodo)


def test_todos_los_dias_tienen_variables_binarias(
    calendario_generado,
):

    binarias = [
        "es_festivo",
        "es_vacaciones",
        "fin_de_semana",
        "restaurante_cerrado",
        "es_vispera_festivo",
        "es_puente",
    ]

    for columna in binarias:

        assert (
            calendario_generado[columna].isin([0, 1]).all()
        ), f"{columna} tiene valores fuera de 0/1"
