# tests/test_modelos.py

# ============================================================
# TESTS DE LOS MODELOS Y DE LA PREDICCIÓN
# ============================================================
#
# Estos tests no comprueban que el modelo sea bueno: eso lo
# dice la evaluación. Comprueban que el sistema se comporta
# como debe:
#
#   - Que las métricas se calculan bien.
#   - Que ningún modelo es peor que el baseline.
#   - Que la predicción responde a las variables de la forma
#     esperada por el negocio.
#   - Que la recomendación de plantilla es coherente.
#
# ============================================================

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COMPARACION_MODELOS,
    EMPLEADOS_MAXIMO,
    EMPLEADOS_MINIMO,
    ruta_modelo,
)


# ============================================================
# MÉTRICAS
# ============================================================

def test_metricas_de_una_prediccion_perfecta():

    from src.models.entrenamiento import calcular_metricas

    reales = np.array([10.0, 20.0, 30.0])

    metricas = calcular_metricas(reales, reales)

    assert metricas["mae"] == 0.0
    assert metricas["rmse"] == 0.0
    assert metricas["mape"] == 0.0
    assert metricas["r2"] == 1.0


def test_metricas_conocidas():

    from src.models.entrenamiento import calcular_metricas

    reales = np.array([100.0, 100.0])
    predichos = np.array([90.0, 110.0])

    metricas = calcular_metricas(reales, predichos)

    assert metricas["mae"] == pytest.approx(10.0)
    assert metricas["rmse"] == pytest.approx(10.0)
    assert metricas["mape"] == pytest.approx(10.0)


def test_el_rmse_penaliza_mas_los_errores_grandes():

    from src.models.entrenamiento import calcular_metricas

    reales = np.array([100.0, 100.0, 100.0, 100.0])

    repartido = np.array([95.0, 95.0, 105.0, 105.0])

    concentrado = np.array([100.0, 100.0, 100.0, 80.0])

    metricas_repartido = calcular_metricas(reales, repartido)
    metricas_concentrado = calcular_metricas(reales, concentrado)

    assert (
        metricas_repartido["mae"]
        == metricas_concentrado["mae"]
    )

    assert (
        metricas_concentrado["rmse"]
        > metricas_repartido["rmse"]
    )


def test_intervalo_de_prediccion():

    from src.models.entrenamiento import calcular_intervalo

    residuos = np.arange(-50, 51, dtype=float)

    inferior, superior = calcular_intervalo(
        residuos,
        nivel=0.80,
    )

    assert inferior < 0 < superior

    assert inferior == pytest.approx(-40, abs=1)

    assert superior == pytest.approx(40, abs=1)


# ============================================================
# BASELINE
# ============================================================

def test_el_baseline_predice_la_media_del_dia():

    from src.models.baseline import MediaPorDiaSemana

    X = pd.DataFrame({
        "dia_semana": ["5", "5", "1", "1"],
    })

    y = pd.Series([100.0, 200.0, 40.0, 60.0])

    modelo = MediaPorDiaSemana().fit(X, y)

    predicciones = modelo.predict(X)

    assert predicciones.tolist() == [150.0, 150.0, 50.0, 50.0]


def test_el_baseline_soporta_dias_desconocidos():

    from src.models.baseline import MediaPorDiaSemana

    X = pd.DataFrame({"dia_semana": ["5", "1"]})

    y = pd.Series([100.0, 50.0])

    modelo = MediaPorDiaSemana().fit(X, y)

    nuevo = pd.DataFrame({"dia_semana": ["3"]})

    assert modelo.predict(nuevo)[0] == pytest.approx(75.0)


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

@pytest.fixture(scope="module")
def comparacion():

    if not COMPARACION_MODELOS.exists():

        pytest.skip(
            "No existe la comparación de modelos. Ejecuta: "
            "python pipeline.py"
        )

    return pd.read_csv(COMPARACION_MODELOS)


def test_se_han_entrenado_los_cuatro_modelos(comparacion):

    esperados = {
        "baseline",
        "regresion_lineal",
        "random_forest",
        "hist_gradient_boosting",
    }

    assert esperados <= set(comparacion["nombre_tecnico"])


def test_todos_los_modelos_mejoran_al_baseline(comparacion):
    """
    Si un modelo de machine learning no supera a "la media de
    los sábados", no aporta nada.
    """

    baseline = comparacion[
        comparacion["nombre_tecnico"] == "baseline"
    ].iloc[0]

    resto = comparacion[
        comparacion["nombre_tecnico"] != "baseline"
    ]

    assert (
        resto["mae_validacion"] < baseline["mae_validacion"]
    ).all()

    assert (resto["mae_test"] < baseline["mae_test"]).all()


def test_las_metricas_son_coherentes(comparacion):
    """
    El RMSE siempre es mayor o igual que el MAE, y el R² de un
    modelo útil es positivo.
    """

    assert (
        comparacion["rmse_validacion"]
        >= comparacion["mae_validacion"]
    ).all()

    assert (comparacion["r2_test"] > 0).all()


# ============================================================
# PREDICCIÓN
# ============================================================

@pytest.fixture(scope="module")
def recursos():

    if not ruta_modelo("produccion").exists():

        pytest.skip(
            "No existe el modelo de producción. Ejecuta: "
            "python pipeline.py"
        )

    from src.models.prediccion import (
        cargar_calendario,
        cargar_modelo_produccion,
    )

    modelo, metadatos = cargar_modelo_produccion()

    return modelo, metadatos, cargar_calendario()


def predecir_con(recursos, **kwargs):

    from src.models.prediccion import predecir

    modelo, metadatos, calendario = recursos

    return predecir(
        modelo=modelo,
        metadatos=metadatos,
        calendario=calendario,
        **kwargs,
    )


def test_la_prediccion_nunca_es_negativa(recursos):
    """
    Ni con las condiciones más adversas posibles.
    """

    resultado = predecir_con(
        recursos,
        fecha=date(2026, 2, 4),
        tmax=-5,
        tmin=-15,
        prec=60,
    )

    assert resultado["prediccion"] >= 0


def test_el_intervalo_contiene_la_prediccion(recursos):

    resultado = predecir_con(
        recursos,
        fecha=date(2026, 6, 20),
        tmax=26,
        tmin=14,
        prec=0,
    )

    assert (
        resultado["intervalo_inferior"]
        <= resultado["prediccion"]
        <= resultado["intervalo_superior"]
    )


def test_el_sabado_tiene_mas_demanda_que_el_miercoles(
    recursos,
):
    """
    Mismas condiciones meteorológicas, distinto día.
    """

    sabado = predecir_con(
        recursos,
        fecha=date(2026, 6, 20),
        tmax=26,
        tmin=14,
        prec=0,
    )

    miercoles = predecir_con(
        recursos,
        fecha=date(2026, 6, 17),
        tmax=26,
        tmin=14,
        prec=0,
    )

    assert sabado["prediccion"] > miercoles["prediccion"]


def test_la_lluvia_reduce_la_prediccion(recursos):
    """
    La hipótesis central del proyecto, comprobada sobre el
    modelo entrenado.
    """

    seco = predecir_con(
        recursos,
        fecha=date(2026, 6, 20),
        tmax=26,
        tmin=14,
        prec=0,
    )

    lluvioso = predecir_con(
        recursos,
        fecha=date(2026, 6, 20),
        tmax=26,
        tmin=14,
        prec=25,
    )

    assert lluvioso["prediccion"] < seco["prediccion"]


def test_los_dias_de_cierre_se_identifican(recursos):

    lunes = predecir_con(
        recursos,
        fecha=date(2026, 6, 15),
        tmax=26,
        tmin=14,
        prec=0,
    )

    assert lunes["cerrado"] is True

    assert lunes["motivo_cierre"]


def test_funciona_fuera_del_calendario_generado(recursos):
    """
    La aplicación no puede quedarse sin respuesta porque el
    usuario elija una fecha lejana.
    """

    resultado = predecir_con(
        recursos,
        fecha=date(2029, 7, 14),
        tmax=30,
        tmin=18,
        prec=0,
    )

    assert resultado["prediccion"] > 0

    assert resultado["en_calendario"] is False


# ============================================================
# RECOMENDACIÓN DE PLANTILLA
# ============================================================

def test_la_plantilla_crece_con_la_demanda():

    from src.models.prediccion import recomendar_empleados

    assert (
        recomendar_empleados(40)
        <= recomendar_empleados(120)
        <= recomendar_empleados(200)
    )


def test_la_plantilla_respeta_los_limites_del_local():

    from src.models.prediccion import recomendar_empleados

    assert recomendar_empleados(0) == EMPLEADOS_MINIMO

    assert recomendar_empleados(10_000) == EMPLEADOS_MAXIMO


def test_evaluar_plantilla_detecta_la_falta_de_personal():

    from src.models.prediccion import evaluar_plantilla

    estado, _, _ = evaluar_plantilla(200, 3)

    assert estado == "riesgo"

    estado, _, _ = evaluar_plantilla(30, 8)

    assert estado == "holgado"


def test_la_plantilla_recomendada_nunca_deja_el_local_en_riesgo():
    """
    La recomendación debe dejar el servicio, como mucho, en
    "personal ocupado", nunca en "falta de personal".
    """

    from src.models.prediccion import (
        evaluar_plantilla,
        recomendar_empleados,
    )

    for clientes in range(20, 180, 10):

        empleados = recomendar_empleados(clientes)

        estado, _, _ = evaluar_plantilla(clientes, empleados)

        assert estado != "riesgo", (
            f"{clientes} clientes con {empleados} empleados "
            "deja el local en riesgo"
        )
