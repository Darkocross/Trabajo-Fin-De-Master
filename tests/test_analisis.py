# tests/test_analisis.py

# ============================================================
# TESTS DE LOS ANÁLISIS DE VALOR Y VALIDACIÓN
# ============================================================
#
# La evaluación económica produce la cifra más citada del
# trabajo ("el sistema ahorra X € al año"). Un error en esta
# lógica no da ningún fallo visible: simplemente sale un
# número equivocado y se defiende con él.
#
# ============================================================

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COSTE_JORNADA_EMPLEADO,
    EMPLEADOS_MAXIMO,
    MARGEN_POR_CLIENTE,
    RATIO_FALTA_PERSONAL,
)

from src.analysis.evaluacion_decision import (
    coste_infradotacion,
    coste_sobredotacion,
    evaluar_metodo,
    plantilla_maxima,
    plantilla_minima,
)

from src.analysis.robustez_prevision import (
    degradar_meteorologia,
)


# ============================================================
# BANDAS DE PLANTILLA
# ============================================================

def test_la_banda_correcta_esta_bien_ordenada():
    """
    La plantilla mínima nunca puede superar a la máxima.
    """

    clientes = np.arange(20, 300, 5)

    assert (
        plantilla_minima(clientes) <= plantilla_maxima(clientes)
    ).all()


def test_mas_clientes_exige_mas_plantilla():

    assert (
        plantilla_minima([50])[0] <= plantilla_minima([150])[0]
    )


# ============================================================
# COSTE DE QUEDARSE CORTO
# ============================================================

def test_no_hay_coste_si_la_plantilla_cubre_la_demanda():
    """
    Con capacidad de sobra no se pierde ningún cliente.
    """

    # 5 empleados cubren 5 x 24 = 120 clientes.
    coste = coste_infradotacion([100], [5])

    assert coste[0] == 0.0


def test_el_coste_crece_con_la_demanda_no_atendida():

    poco = coste_infradotacion([130], [5])[0]
    mucho = coste_infradotacion([200], [5])[0]

    assert 0 < poco < mucho


def test_el_coste_de_quedarse_corto_esta_bien_calculado():
    """
    Con 5 empleados la capacidad es 120. Con 140 clientes
    sobran 20, de los que se pierde la mitad.
    """

    coste = coste_infradotacion(
        [140],
        [5],
        fraccion_perdida=0.5,
        margen=10.0,
    )[0]

    assert coste == pytest.approx(20 * 0.5 * 10.0)


# ============================================================
# COSTE DE PASARSE
# ============================================================

def test_no_hay_coste_si_la_plantilla_es_razonable():

    # 100 clientes admiten hasta ceil(100/15) = 7 empleados.
    assert coste_sobredotacion([100], [7])[0] == 0.0


def test_cada_empleado_de_mas_cuesta_una_jornada():

    coste = coste_sobredotacion([100], [9])[0]

    assert coste == pytest.approx(2 * COSTE_JORNADA_EMPLEADO)


# ============================================================
# EVALUACIÓN DE UN MÉTODO
# ============================================================

def test_la_prediccion_perfecta_no_tiene_coste():
    """
    Si se conocen los clientes exactos, la plantilla
    recomendada debe caer siempre dentro de la banda correcta.
    """

    reales = np.array(
        [40, 60, 90, 120, 150, 180],
        dtype=float,
    )

    resultado = evaluar_metodo(reales, reales)

    assert resultado["correcto"].all()

    assert resultado["coste_total"].sum() == 0.0


def test_cada_dia_cae_en_una_sola_categoria():

    reales = np.array([50, 120, 200], dtype=float)
    estimados = np.array([200, 120, 50], dtype=float)

    resultado = evaluar_metodo(reales, estimados)

    suma = (
        resultado["correcto"].astype(int)
        + resultado["infradotado"].astype(int)
        + resultado["sobredotado"].astype(int)
    )

    assert (suma == 1).all()


def test_subestimar_mucho_deja_el_local_corto():
    """
    Predecir 40 clientes cuando vienen 250 tiene que salir
    como falta de personal.
    """

    resultado = evaluar_metodo([250.0], [40.0])

    assert resultado["infradotado"][0]

    assert resultado["coste_total"][0] > 0


def test_sobrestimar_mucho_deja_personal_de_sobra():

    resultado = evaluar_metodo([40.0], [250.0])

    assert resultado["sobredotado"][0]


# ============================================================
# DEGRADACIÓN DE LA METEOROLOGÍA
# ============================================================

@pytest.fixture
def meteorologia():

    return pd.DataFrame({
        "tmax": [25.0, 30.0, 12.0, 18.0],
        "tmin": [15.0, 18.0, 2.0, 9.0],
        "tmed": [20.0, 24.0, 7.0, 13.5],
        "prec": [0.0, 0.0, 12.0, 3.0],
    })


def test_sin_error_la_meteorologia_no_cambia(meteorologia):
    """
    El escenario de referencia debe dejar los datos intactos.
    """

    rng = np.random.default_rng(0)

    degradado = degradar_meteorologia(
        meteorologia,
        sigma_temperatura=0.0,
        prob_fallo_lluvia=0.0,
        rng=rng,
    )

    for columna in ("tmax", "tmin", "tmed", "prec"):

        assert (
            degradado[columna].to_numpy()
            == meteorologia[columna].to_numpy()
        ).all()


def test_la_degradacion_mantiene_tmin_por_debajo_de_tmax(
    meteorologia,
):
    """
    Por mucho ruido que se meta, no puede salir un día con la
    mínima por encima de la máxima.
    """

    for semilla in range(20):

        rng = np.random.default_rng(semilla)

        degradado = degradar_meteorologia(
            meteorologia,
            sigma_temperatura=4.0,
            prob_fallo_lluvia=0.4,
            rng=rng,
        )

        assert (
            degradado["tmin"] <= degradado["tmax"]
        ).all()


def test_la_degradacion_no_inventa_lluvia_negativa(
    meteorologia,
):

    rng = np.random.default_rng(7)

    degradado = degradar_meteorologia(
        meteorologia,
        sigma_temperatura=3.0,
        prob_fallo_lluvia=0.35,
        rng=rng,
    )

    assert (degradado["prec"] >= 0).all()


def test_mas_error_significa_mas_desviacion(meteorologia):
    """
    Comprobación de que el parámetro hace lo que dice.
    """

    def desviacion(sigma):

        errores = []

        for semilla in range(60):

            rng = np.random.default_rng(semilla)

            degradado = degradar_meteorologia(
                meteorologia,
                sigma_temperatura=sigma,
                prob_fallo_lluvia=0.0,
                rng=rng,
            )

            errores.append(
                (
                    degradado["tmax"] - meteorologia["tmax"]
                ).abs().mean()
            )

        return float(np.mean(errores))

    assert desviacion(1.0) < desviacion(3.0)


# ============================================================
# COHERENCIA CON LOS RESULTADOS PUBLICADOS
# ============================================================

def test_el_modelo_decide_mejor_que_la_intuicion():
    """
    Es la afirmación central del trabajo. Si dejara de
    cumplirse, habría que reescribir las conclusiones.
    """

    from src.config import PROCESSED_RESULTADOS_DIR

    ruta = PROCESSED_RESULTADOS_DIR / "resumen_decision.csv"

    if not ruta.exists():

        pytest.skip(
            "No existe el resumen de decisión. Ejecuta: "
            "python pipeline.py"
        )

    resumen = pd.read_csv(ruta).set_index("metodo")

    assert (
        resumen.loc["Modelo", "dias_correctos"]
        > resumen.loc["Intuición", "dias_correctos"]
    )

    assert (
        resumen.loc["Modelo", "coste_total"]
        < resumen.loc["Intuición", "coste_total"]
    )

    # El oráculo es el suelo: nadie puede hacerlo mejor.
    assert (
        resumen.loc["Oráculo", "coste_total"]
        <= resumen.loc["Modelo", "coste_total"]
    )


def test_el_ahorro_no_depende_de_los_supuestos():
    """
    El análisis de sensibilidad debe salir favorable en todos
    los escenarios; si no, la cifra de ahorro no se sostiene.
    """

    from src.config import PROCESSED_RESULTADOS_DIR

    ruta = PROCESSED_RESULTADOS_DIR / "sensibilidad_costes.csv"

    if not ruta.exists():

        pytest.skip("Ejecuta antes: python pipeline.py")

    sensibilidad = pd.read_csv(ruta)

    assert sensibilidad["modelo_mejor"].all(), (
        "El modelo deja de ser mejor en algún escenario: "
        "revisa la conclusión económica"
    )


# ============================================================
# INTERVALO ESCALADO
# ============================================================

def test_el_intervalo_se_ensancha_con_la_prediccion():
    """
    Un sábado lleno tiene más incertidumbre que un martes
    flojo, y el intervalo tiene que reflejarlo.
    """

    import numpy as np

    from src.models.entrenamiento import (
        aplicar_intervalo,
        calcular_intervalo_escalado,
    )

    rng = np.random.default_rng(0)

    predicciones = rng.uniform(40, 200, 400)

    # Error proporcional al nivel: es lo que pasa en los datos.
    residuos = rng.normal(0, 1, 400) * predicciones * 0.2

    inferior, superior = calcular_intervalo_escalado(
        predicciones,
        residuos,
    )

    def anchura(valor):
        return (
            aplicar_intervalo(valor, superior)
            - aplicar_intervalo(valor, inferior)
        )

    assert anchura(180) > anchura(60)


def test_con_pocos_datos_cae_al_intervalo_fijo():
    """
    Ajustar una recta con veinte puntos es peor que no
    ajustarla. En ese caso se usa la constante de siempre.
    """

    import numpy as np

    from src.models.entrenamiento import (
        calcular_intervalo_escalado,
    )

    rng = np.random.default_rng(0)

    inferior, superior = calcular_intervalo_escalado(
        rng.uniform(40, 200, 20),
        rng.normal(0, 10, 20),
    )

    # Pendiente cero = intervalo constante.
    assert inferior[1] == 0.0
    assert superior[1] == 0.0


# ============================================================
# MONITORIZACIÓN
# ============================================================

def test_la_deriva_sostenida_dispara_el_reentrenamiento():
    """
    Varias ventanas seguidas fallando en la misma dirección y
    por encima de medio empleado: hay que reentrenar.
    """

    import pandas as pd

    from src.analysis.monitorizacion import diagnostico

    resumen = pd.DataFrame({
        "sesgo": [12.0, 14.0, 13.0],
        "dias": [20, 20, 20],
        "fiable": [True, True, True],
    })

    hay_que_reentrenar, motivo = diagnostico(resumen)

    assert hay_que_reentrenar
    assert "corto" in motivo


def test_el_ruido_no_dispara_el_reentrenamiento():
    """
    Ventanas grandes pero alternando de signo: es ruido. Si
    esto disparase una alerta, nadie volvería a hacerle caso.
    """

    import pandas as pd

    from src.analysis.monitorizacion import diagnostico

    resumen = pd.DataFrame({
        "sesgo": [14.0, -13.0, 15.0],
        "dias": [20, 20, 20],
        "fiable": [True, True, True],
    })

    hay_que_reentrenar, _ = diagnostico(resumen)

    assert not hay_que_reentrenar


def test_las_ventanas_cortas_no_cuentan():
    """
    El local cierra lunes, martes y todo agosto. Las ventanas
    que caen sobre un cierre quedan con tres o cuatro días y su
    media no significa nada.
    """

    import pandas as pd

    from src.analysis.monitorizacion import diagnostico

    resumen = pd.DataFrame({
        "sesgo": [40.0, 45.0, 38.0],
        "dias": [3, 2, 4],
        "fiable": [False, False, False],
    })

    hay_que_reentrenar, motivo = diagnostico(resumen)

    assert not hay_que_reentrenar
    assert "fiables" in motivo


# ============================================================
# PREVISIÓN DE LOS PRÓXIMOS DÍAS
# ============================================================

def test_el_grafico_de_prevision_se_construye():
    """
    El gráfico tiene que aguantar una semana con días de cierre
    intercalados, que es el caso normal: el local cierra lunes
    y martes.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd

    from src.visualization.prevision_semana import (
        grafico_prevision,
    )

    prevision = pd.DataFrame([
        {
            "fecha": "2026-09-14", "dia": "Lunes", "abierto": 0,
            "motivo_cierre": "Descanso del personal",
            "tmax": 30.0, "tmin": 15.0, "prob_lluvia": 0.0,
            "clientes_estimados": None,
            "intervalo_inferior": None,
            "intervalo_superior": None,
            "empleados_recomendados": 0,
            "media_historica": None,
        },
        {
            "fecha": "2026-09-15", "dia": "Martes", "abierto": 1,
            "motivo_cierre": "",
            "tmax": 28.0, "tmin": 14.0, "prob_lluvia": 20.0,
            "clientes_estimados": 70.0,
            "intervalo_inferior": 54.0,
            "intervalo_superior": 102.0,
            "empleados_recomendados": 4,
            "media_historica": 66.0,
        },
    ])

    figura = grafico_prevision(prevision, titulo="Prueba")

    ejes = figura.axes[0]

    # Una sola barra: el día cerrado no tiene.
    barras = [
        p for p in ejes.patches
        if getattr(p, "get_height", None)
        and p.get_height() == 70.0
    ]

    assert len(barras) == 1

    # Y un solo eje: nada de doble escala.
    assert len(figura.axes) == 1

    plt.close(figura)


def test_el_grafico_aguanta_una_semana_entera_cerrada():
    """
    En agosto el local cierra el mes completo. El gráfico no
    puede romperse por no tener ninguna barra que dibujar.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd

    from src.visualization.prevision_semana import (
        grafico_prevision,
    )

    prevision = pd.DataFrame([
        {
            "fecha": f"2026-08-0{dia}", "dia": "Lunes",
            "abierto": 0, "motivo_cierre": "Vacaciones",
            "tmax": 35.0, "tmin": 20.0, "prob_lluvia": 0.0,
            "clientes_estimados": None,
            "intervalo_inferior": None,
            "intervalo_superior": None,
            "empleados_recomendados": 0,
            "media_historica": None,
        }
        for dia in (1, 2, 3)
    ])

    figura = grafico_prevision(prevision)

    assert figura is not None

    plt.close(figura)
