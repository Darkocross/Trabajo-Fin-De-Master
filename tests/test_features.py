# tests/test_features.py

# ============================================================
# TESTS DEL MÓDULO DE VARIABLES
# ============================================================
#
# `features.py` decide qué ve el modelo. Estos tests fijan por
# escrito las decisiones que se tomaron y por qué, de forma que
# nadie pueda deshacerlas sin darse cuenta:
#
#   - Que no entra ninguna variable de resultado (leakage).
#   - Que no vuelven las variables colineales.
#   - Que las derivadas se calculan igual en entrenamiento y en
#     predicción.
#   - Que la separación temporal no mezcla periodos.
#
# ============================================================

import numpy as np
import pandas as pd
import pytest

from src.config import AÑOS_TRAIN, AÑO_TEST, AÑO_VALIDACION

from src.models.features import (
    VARIABLE_OBJETIVO,
    VARIABLES,
    VARIABLES_CATEGORICAS,
    VARIABLES_NUMERICAS,
    anadir_variables_derivadas,
    crear_preprocesador,
    preparar_dataset,
    separar_temporal,
    separar_x_y,
    traducir_variable,
    validar_separacion,
)


# ============================================================
# COMPOSICIÓN DEL CONJUNTO DE VARIABLES
# ============================================================

def test_no_hay_variables_repetidas():

    assert len(VARIABLES) == len(set(VARIABLES))


def test_la_variable_objetivo_no_es_una_entrada():
    """
    Predecir usando lo que se quiere predecir sería el error
    más grave posible.
    """

    assert VARIABLE_OBJETIVO not in VARIABLES


def test_no_entran_variables_de_resultado():
    """
    `nota_faena` describe cómo fue la jornada y `n_empleados`
    se decide a partir de la demanda esperada. Ninguna de las
    dos está disponible de forma honesta antes del servicio.
    """

    prohibidas = ["nota_faena", "n_empleados", "n_clientes"]

    for variable in prohibidas:

        assert variable not in VARIABLES


def test_no_vuelven_las_variables_colineales():
    """
    Estas variables son función exacta de otras que ya están:

        fin_de_semana = dia_semana in {5, 6}
        es_vacaciones = tipo_vacaciones != "ninguna"
        tmax, tmin    = tmed +- amplitud/2

    Y estas ya no existen en ninguna capa: eran hipótesis de
    modelado disfrazadas de datos.

        es_semana_santa, es_mayo, es_junio, es_julio
        es_fin_semana_*

    Meterlas de nuevo rompería la interpretabilidad de la
    regresión lineal.
    """

    colineales = [
        "fin_de_semana",
        "es_vacaciones",
        "es_semana_santa",
        "es_mayo",
        "es_junio",
        "es_julio",
        "es_fin_semana_semana_santa",
        "es_fin_semana_mayo",
        "es_fin_semana_junio",
        "es_fin_semana_julio",
        "tmax",
        "tmin",
    ]

    for variable in colineales:

        assert variable not in VARIABLES, (
            f"'{variable}' es colineal con otra variable ya "
            "presente"
        )


def test_las_categoricas_y_numericas_no_se_solapan():

    assert not (
        set(VARIABLES_CATEGORICAS)
        & set(VARIABLES_NUMERICAS)
    )


# ============================================================
# VARIABLES DERIVADAS
# ============================================================

def test_amplitud_termica():

    datos = pd.DataFrame({
        "tmax": [25.0, 10.0],
        "tmin": [15.0, -2.0],
        "prec": [0.0, 0.0],
        "fin_de_semana": [0, 1],
    })

    resultado = anadir_variables_derivadas(datos)

    assert resultado["amplitud_termica"].tolist() == [10.0, 12.0]


def test_precipitacion_logaritmica():

    datos = pd.DataFrame({
        "tmax": [20.0],
        "tmin": [10.0],
        "prec": [9.0],
        "fin_de_semana": [0],
    })

    resultado = anadir_variables_derivadas(datos)

    assert resultado["prec_log"].iloc[0] == pytest.approx(
        np.log1p(9.0)
    )


def test_lluvia_solo_interactua_en_fin_de_semana():
    """
    La interacción debe ser cero entre semana, sea cual sea la
    lluvia.
    """

    datos = pd.DataFrame({
        "tmax": [20.0, 20.0],
        "tmin": [10.0, 10.0],
        "prec": [12.0, 12.0],
        "fin_de_semana": [0, 1],
    })

    resultado = anadir_variables_derivadas(datos)

    assert resultado["lluvia_fin_semana"].iloc[0] == 0.0

    assert resultado["lluvia_fin_semana"].iloc[1] > 0.0


def test_precipitacion_negativa_se_corrige():
    """
    Un valor negativo de lluvia no tiene sentido físico y
    rompería el logaritmo.
    """

    datos = pd.DataFrame({
        "tmax": [20.0],
        "tmin": [10.0],
        "prec": [-3.0],
        "fin_de_semana": [0],
    })

    resultado = anadir_variables_derivadas(datos)

    assert resultado["prec_log"].iloc[0] == 0.0


# ============================================================
# PREPARACIÓN DEL DATASET
# ============================================================

def test_preparar_dataset_falla_si_faltan_columnas():

    with pytest.raises(ValueError, match="Faltan columnas"):

        preparar_dataset(pd.DataFrame({"fecha": []}))


def test_preparar_dataset_produce_todas_las_variables(gold):

    preparado = preparar_dataset(gold)

    for variable in VARIABLES:
        assert variable in preparado.columns


def test_las_categoricas_se_tratan_como_texto(gold):
    """
    `dia_semana` y `mes` son números, pero el efecto del
    sábado no es "seis veces" el del lunes. Deben llegar al
    One-Hot como categorías.
    """

    preparado = preparar_dataset(gold)

    for columna in VARIABLES_CATEGORICAS:

        # No se compara con un dtype concreto porque cambia
        # entre versiones de pandas ("object" o "str"). Lo que
        # importa es que NO sea numérico.
        assert not pd.api.types.is_numeric_dtype(
            preparado[columna]
        ), f"'{columna}' debe llegar al One-Hot como categoría"

        assert isinstance(preparado[columna].iloc[0], str)


def test_separar_x_y_respeta_el_orden(gold):

    preparado = preparar_dataset(gold)

    X, y = separar_x_y(preparado)

    assert list(X.columns) == VARIABLES

    assert len(X) == len(y)

    assert y.name == VARIABLE_OBJETIVO


# ============================================================
# SEPARACIÓN TEMPORAL
# ============================================================

def test_separacion_por_años(gold):

    train, validacion, test = separar_temporal(gold)

    assert set(train["fecha"].dt.year.unique()) <= set(
        AÑOS_TRAIN
    )

    assert (
        validacion["fecha"].dt.year == AÑO_VALIDACION
    ).all()

    assert (test["fecha"].dt.year >= AÑO_TEST).all()


def test_los_conjuntos_no_se_solapan(gold):

    train, validacion, test = separar_temporal(gold)

    validar_separacion(train, validacion, test)

    assert train["fecha"].max() < validacion["fecha"].min()

    assert validacion["fecha"].max() < test["fecha"].min()


def test_la_separacion_no_pierde_registros(gold):

    train, validacion, test = separar_temporal(gold)

    assert len(train) + len(validacion) + len(test) == len(gold)


def test_validar_separacion_detecta_conjuntos_vacios(gold):

    train, validacion, test = separar_temporal(gold)

    with pytest.raises(ValueError, match="vacío"):

        validar_separacion(train.head(0), validacion, test)


# ============================================================
# PREPROCESADOR
# ============================================================

def test_el_preprocesador_no_deja_nulos(gold):

    preparado = preparar_dataset(gold)

    X, y = separar_x_y(preparado)

    # Se introduce un hueco a propósito.
    X = X.copy()
    X.loc[X.index[0], "tmed"] = np.nan

    preprocesador = crear_preprocesador(escalar=False)

    transformado = preprocesador.fit_transform(X)

    assert not np.isnan(transformado).any()


def test_el_escalado_solo_afecta_a_las_numericas(gold):
    """
    La regresión lineal necesita variables numéricas
    estandarizadas para que sus coeficientes sean comparables.
    Las categóricas, ya codificadas como 0/1, no deben tocarse.
    """

    preparado = preparar_dataset(gold)

    X, y = separar_x_y(preparado)

    escalado = crear_preprocesador(escalar=True)

    transformado = escalado.fit_transform(X)

    nombres = escalado.get_feature_names_out()

    for indice, nombre in enumerate(nombres):

        columna = transformado[:, indice]

        if nombre.startswith("categoricas__"):

            assert set(np.unique(columna)) <= {0.0, 1.0}

        elif nombre.startswith("numericas__"):

            # Media prácticamente cero tras estandarizar.
            assert abs(columna.mean()) < 1e-6


def test_el_onehot_no_falla_con_categorias_nuevas(gold):
    """
    En producción puede aparecer un valor que no estaba en el
    entrenamiento. El modelo debe seguir respondiendo.
    """

    preparado = preparar_dataset(gold)

    X, y = separar_x_y(preparado)

    preprocesador = crear_preprocesador(escalar=False)

    preprocesador.fit(X)

    nuevo = X.head(1).copy()

    nuevo.loc[nuevo.index[0], "tipo_vacaciones"] = (
        "categoria_desconocida"
    )

    transformado = preprocesador.transform(nuevo)

    assert transformado.shape[0] == 1


# ============================================================
# NOMBRES LEGIBLES
# ============================================================

@pytest.mark.parametrize(
    "tecnico,legible",
    [
        ("categoricas__dia_semana_5", "día de la semana: sábado"),
        ("categoricas__mes_12", "mes: diciembre"),
        ("numericas__tmed", "temperatura media"),
        (
            "numericas__amplitud_termica",
            "amplitud térmica",
        ),
        (
            "categoricas__tipo_vacaciones_navidad",
            "tipo de vacaciones: navidad",
        ),
    ],
)
def test_traducir_variable(tecnico, legible):

    assert traducir_variable(tecnico) == legible
