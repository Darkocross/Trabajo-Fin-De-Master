# src/models/entrenamiento.py

# ============================================================
# PROTOCOLO COMÚN DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================
#
# Todos los modelos del proyecto se entrenan y se evalúan
# exactamente igual:
#
#   1. Se cargan los mismos conjuntos train / validación / test.
#   2. Se usan las mismas variables (`src/models/features.py`).
#   3. Se calculan las mismas métricas.
#   4. Se guardan el modelo, sus predicciones y sus métricas
#      con la misma estructura.
#
# Gracias a esto la comparación entre modelos es justa: la
# única diferencia entre ellos es el algoritmo.
#
# ============================================================

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    COMPARACION_MODELOS,
    METADATOS_MODELOS,
    MODELOS_DIR,
    PROCESSED_RESULTADOS_DIR,
    TEST,
    TRAIN,
    VALIDATION,
    ruta_modelo,
)

from src.models.features import (
    VARIABLE_OBJETIVO,
    VARIABLES,
    preparar_dataset,
    separar_x_y,
    validar_separacion,
)


# ============================================================
# CARGA DE LOS CONJUNTOS
# ============================================================

def cargar_conjunto(ruta, nombre):
    """
    Carga uno de los conjuntos de modelado y lo deja listo
    para el modelo.
    """

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existe el conjunto {nombre}:\n{ruta}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py --hasta separar"
        )

    df = pd.read_csv(
        ruta,
        parse_dates=["fecha"],
    )

    df = preparar_dataset(df)

    return df


def cargar_conjuntos(verbose=True):
    """
    Carga train, validación y test, comprueba que la
    separación temporal es correcta y devuelve los tres.
    """

    train = cargar_conjunto(TRAIN, "TRAIN")
    validacion = cargar_conjunto(VALIDATION, "VALIDATION")
    test = cargar_conjunto(TEST, "TEST")

    validar_separacion(train, validacion, test)

    if verbose:

        print()
        print("CONJUNTOS DE MODELADO")
        print("-" * 60)

        for nombre, conjunto in [
            ("Train", train),
            ("Validación", validacion),
            ("Test", test),
        ]:

            print(
                f"{nombre:<12} "
                f"{len(conjunto):>5} registros   "
                f"{conjunto['fecha'].min().date()} -> "
                f"{conjunto['fecha'].max().date()}"
            )

        print()
        print(f"Variables de entrada: {len(VARIABLES)}")
        print(f"Variable objetivo:    {VARIABLE_OBJETIVO}")

    return train, validacion, test


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(y_real, y_pred):
    """
    Calcula las métricas de error del proyecto.

    - MAE:   error medio en número de clientes. Es la métrica
             principal porque se interpreta directamente en
             las unidades del negocio.
    - RMSE:  penaliza más los errores grandes. Sirve para
             detectar días en los que el modelo falla mucho.
    - MAPE:  error relativo. Permite comparar días de mucha y
             de poca afluencia.
    - R²:    proporción de variabilidad explicada.
    """

    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_real, y_pred)

    rmse = float(
        np.sqrt(
            mean_squared_error(y_real, y_pred)
        )
    )

    # Se evita dividir por cero en días sin clientes.
    mascara = y_real > 0

    if mascara.any():

        mape = float(
            np.mean(
                np.abs(
                    (y_real[mascara] - y_pred[mascara])
                    / y_real[mascara]
                )
            )
            * 100
        )

    else:

        mape = float("nan")

    r2 = float(r2_score(y_real, y_pred))

    return {
        "mae": float(mae),
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def mostrar_metricas(titulo, metricas):
    """
    Imprime un bloque de métricas con formato uniforme.
    """

    print()
    print(titulo)
    print("-" * 60)
    print(f"MAE:   {metricas['mae']:.2f} clientes")
    print(f"RMSE:  {metricas['rmse']:.2f} clientes")
    print(f"MAPE:  {metricas['mape']:.1f} %")
    print(f"R²:    {metricas['r2']:.3f}")


# ============================================================
# INTERVALO DE PREDICCIÓN
# ============================================================

def calcular_intervalo(residuos, nivel=0.80):
    """
    Calcula un intervalo de predicción empírico a partir de los
    residuos del conjunto de validación.

    No se asume que el error siga una distribución normal: se
    toman directamente los percentiles de los residuos
    observados. Es un método sencillo, honesto y suficiente
    para el MVP.

    Devuelve el desplazamiento inferior y superior que hay que
    sumar a la predicción puntual.
    """

    residuos = np.asarray(residuos, dtype=float)
    residuos = residuos[~np.isnan(residuos)]

    if residuos.size == 0:
        return 0.0, 0.0

    alfa = (1 - nivel) / 2

    inferior = float(np.quantile(residuos, alfa))
    superior = float(np.quantile(residuos, 1 - alfa))

    return inferior, superior


def calcular_intervalo_escalado(
    predicciones,
    residuos,
    nivel=0.80,
):
    """
    Intervalo de predicción cuya anchura crece con la propia
    predicción.

    POR QUÉ NO VALE UNO FIJO. El intervalo de percentiles
    constantes reparte la misma incertidumbre a todos los días,
    y el error de este problema no es constante ni de lejos: la
    desviación de los residuos pasa de 11,7 clientes en los
    días flojos a 36,4 en los días fuertes.

    El efecto sobre la cobertura real es el que cabe esperar
    (medido sobre test, intervalo al 80 %):

        predicción      fijo    escalado
        menos de 60    100,0 %    69,0 %
        60-90           76,9 %    71,2 %
        90-120          87,2 %    89,7 %
        120-160         60,6 %    78,8 %
        más de 160      57,1 %    64,3 %

    El intervalo fijo no es que sea impreciso: es que miente en
    las dos direcciones a la vez. Sobra en los días tranquilos,
    donde cubre el 100 % y no informa de nada, y falta justo en
    los días llenos, donde cubre el 57 % y es cuando de verdad
    hace falta saber hasta dónde puede llegar la cosa.

    CÓMO SE HACE. Regresión cuantílica de los residuos sobre la
    predicción: en lugar de un número por cada extremo, una
    recta. La anchura pasa así de unos 25 clientes un martes
    flojo a unos 79 un sábado lleno, con la misma anchura MEDIA
    que antes. No se añade incertidumbre: se reparte donde
    corresponde.

    Devuelve dos pares (a, b), de modo que el intervalo es:

        inferior = predicción + (a_inf + b_inf * predicción)
        superior = predicción + (a_sup + b_sup * predicción)
    """

    from sklearn.linear_model import QuantileRegressor

    predicciones = np.asarray(predicciones, dtype=float)
    residuos = np.asarray(residuos, dtype=float)

    validos = ~(np.isnan(predicciones) | np.isnan(residuos))

    predicciones = predicciones[validos]
    residuos = residuos[validos]

    if predicciones.size < 30:

        # Con pocos datos la recta es peor que la constante.
        inferior, superior = calcular_intervalo(residuos, nivel)

        return (inferior, 0.0), (superior, 0.0)

    alfa = (1 - nivel) / 2

    X = predicciones.reshape(-1, 1)

    recta_inferior = QuantileRegressor(
        quantile=alfa,
        alpha=0.0,
        solver="highs",
    ).fit(X, residuos)

    recta_superior = QuantileRegressor(
        quantile=1 - alfa,
        alpha=0.0,
        solver="highs",
    ).fit(X, residuos)

    return (
        (
            float(recta_inferior.intercept_),
            float(recta_inferior.coef_[0]),
        ),
        (
            float(recta_superior.intercept_),
            float(recta_superior.coef_[0]),
        ),
    )


def aplicar_intervalo(prediccion, coeficientes):
    """
    Aplica los coeficientes del intervalo a una predicción.

    Acepta tanto el formato escalado (a, b) como un número
    suelto, para que los modelos antiguos sigan funcionando.
    """

    if isinstance(coeficientes, (list, tuple)):
        a, b = coeficientes

    else:
        a, b = float(coeficientes), 0.0

    return prediccion + a + b * prediccion


# ============================================================
# GUARDAR PREDICCIONES
# ============================================================

def guardar_predicciones(
    nombre_modelo,
    validacion,
    pred_validacion,
    test,
    pred_test,
):
    """
    Guarda las predicciones de un modelo sobre validación y
    test, con su error, para poder analizarlas después.
    """

    PROCESSED_RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bloques = []

    for conjunto, datos, predicciones in [
        ("validacion", validacion, pred_validacion),
        ("test", test, pred_test),
    ]:

        bloque = datos[
            [
                "fecha",
                "dia_semana",
                "mes",
                "es_festivo",
                "tipo_vacaciones",
                "tmed",
                "tmax",
                "tmin",
                "prec",
                VARIABLE_OBJETIVO,
            ]
        ].copy()

        bloque["conjunto"] = conjunto
        bloque["modelo"] = nombre_modelo
        bloque["prediccion"] = np.round(predicciones, 2)

        bloque["error"] = (
            bloque[VARIABLE_OBJETIVO]
            - bloque["prediccion"]
        )

        bloque["error_absoluto"] = bloque["error"].abs()

        bloques.append(bloque)

    resultados = pd.concat(
        bloques,
        ignore_index=True,
    ).sort_values(
        ["conjunto", "fecha"]
    ).reset_index(drop=True)

    ruta = (
        PROCESSED_RESULTADOS_DIR
        / f"predicciones_{nombre_modelo}.csv"
    )

    resultados.to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )

    return ruta, resultados


# ============================================================
# REGISTRO DE RESULTADOS
# ============================================================

def registrar_resultado(registro):
    """
    Añade o actualiza la fila de un modelo en el fichero de
    comparación de modelos.

    Cada modelo se entrena en su propio script, pero todos
    escriben en la misma tabla, de forma que la comparación
    siempre refleja la última ejecución de cada uno.
    """

    PROCESSED_RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columnas = [
        "modelo",
        "nombre_tecnico",
        "mae_validacion",
        "rmse_validacion",
        "mape_validacion",
        "r2_validacion",
        "mae_test",
        "rmse_test",
        "mape_test",
        "r2_test",
        "entrenado",
    ]

    if COMPARACION_MODELOS.exists():

        comparacion = pd.read_csv(COMPARACION_MODELOS)

        comparacion = comparacion[
            comparacion["nombre_tecnico"]
            != registro["nombre_tecnico"]
        ]

    else:

        comparacion = pd.DataFrame(columns=columnas)

    comparacion = pd.concat(
        [
            comparacion,
            pd.DataFrame([registro]),
        ],
        ignore_index=True,
    )

    comparacion = comparacion[columnas]

    # Se ordena por error en validación: el conjunto que sirve
    # para elegir modelo. El test se reserva como estimación
    # final e imparcial.
    comparacion = comparacion.sort_values(
        "mae_validacion"
    ).reset_index(drop=True)

    numericas = [
        columna
        for columna in columnas
        if columna.startswith(
            ("mae_", "rmse_", "mape_", "r2_")
        )
    ]

    comparacion[numericas] = (
        comparacion[numericas].astype(float).round(3)
    )

    comparacion.to_csv(
        COMPARACION_MODELOS,
        index=False,
        encoding="utf-8",
    )

    return comparacion


def guardar_metadatos(nombre_modelo, metadatos):
    """
    Guarda información sobre el modelo entrenado: variables
    usadas, métricas e intervalo de predicción.

    La aplicación lee este fichero para no tener ninguna cifra
    escrita a mano.
    """

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    if METADATOS_MODELOS.exists():

        with open(
            METADATOS_MODELOS,
            "r",
            encoding="utf-8",
        ) as archivo:

            try:
                todos = json.load(archivo)

            except json.JSONDecodeError:
                todos = {}

    else:

        todos = {}

    todos[nombre_modelo] = metadatos

    with open(
        METADATOS_MODELOS,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            todos,
            archivo,
            ensure_ascii=False,
            indent=4,
        )


def cargar_metadatos(nombre_modelo=None):
    """
    Lee los metadatos guardados de los modelos.
    """

    if not METADATOS_MODELOS.exists():

        raise FileNotFoundError(
            "No existen metadatos de modelos.\n"
            "Ejecuta antes: python pipeline.py"
        )

    with open(
        METADATOS_MODELOS,
        "r",
        encoding="utf-8",
    ) as archivo:

        todos = json.load(archivo)

    if nombre_modelo is None:
        return todos

    if nombre_modelo not in todos:

        raise KeyError(
            f"No hay metadatos del modelo '{nombre_modelo}'."
        )

    return todos[nombre_modelo]


# ============================================================
# ENTRENAMIENTO COMPLETO
# ============================================================

def entrenar_y_evaluar(
    nombre_tecnico,
    nombre_legible,
    modelo,
    train=None,
    validacion=None,
    test=None,
    guardar=True,
):
    """
    Ejecuta el protocolo completo para un modelo:

        entrenar -> predecir -> medir -> guardar

    Devuelve el modelo entrenado y el diccionario de métricas.
    """

    print()
    print("=" * 60)
    print(nombre_legible.upper())
    print("=" * 60)

    # --------------------------------------------------------
    # CONJUNTOS
    # --------------------------------------------------------

    if train is None or validacion is None or test is None:

        train, validacion, test = cargar_conjuntos()

    X_train, y_train = separar_x_y(train)
    X_validacion, y_validacion = separar_x_y(validacion)
    X_test, y_test = separar_x_y(test)

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------

    print()
    print("Entrenando...")

    modelo.fit(X_train, y_train)

    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------
    #
    # El número de clientes no puede ser negativo. Se recorta a
    # cero por coherencia con el dominio del problema.

    pred_validacion = np.clip(
        modelo.predict(X_validacion),
        0,
        None,
    )

    pred_test = np.clip(
        modelo.predict(X_test),
        0,
        None,
    )

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    metricas_validacion = calcular_metricas(
        y_validacion,
        pred_validacion,
    )

    metricas_test = calcular_metricas(
        y_test,
        pred_test,
    )

    mostrar_metricas("VALIDACIÓN (2025)", metricas_validacion)
    mostrar_metricas("TEST (2026)", metricas_test)

    # --------------------------------------------------------
    # INTERVALO DE PREDICCIÓN
    # --------------------------------------------------------
    #
    # Se calcula sobre validación, nunca sobre test: el test
    # debe seguir siendo un conjunto no utilizado.

    residuos_validacion = (
        np.asarray(y_validacion, dtype=float)
        - pred_validacion
    )

    intervalo_inferior, intervalo_superior = calcular_intervalo(
        residuos_validacion,
        nivel=0.80,
    )

    # Intervalo que escala con la predicción. Es el que usa el
    # producto; el fijo se conserva como referencia.

    escalado_inferior, escalado_superior = (
        calcular_intervalo_escalado(
            pred_validacion,
            residuos_validacion,
            nivel=0.80,
        )
    )

    print()
    print("INTERVALO DE PREDICCIÓN (80 %)")
    print("-" * 60)

    print(
        f"Fijo:      {intervalo_inferior:+.0f} / "
        f"{intervalo_superior:+.0f} clientes"
    )

    for referencia in (60, 120, 180):

        inferior = aplicar_intervalo(
            referencia,
            escalado_inferior,
        )

        superior = aplicar_intervalo(
            referencia,
            escalado_superior,
        )

        print(
            f"Escalado:  {referencia:>3} clientes  ->  "
            f"{inferior:.0f} - {superior:.0f}"
            f"   (anchura {superior - inferior:.0f})"
        )

    if not guardar:
        return modelo, {
            "validacion": metricas_validacion,
            "test": metricas_test,
        }

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    ruta_pkl = ruta_modelo(nombre_tecnico)

    # `compress=3` reduce mucho el tamaño del fichero sin
    # alterar el modelo. El Random Forest pasa de 5 MB a menos
    # de 1 MB, lo que hace razonable versionarlo en el
    # repositorio para que la aplicación funcione nada más
    # clonar el proyecto.
    joblib.dump(modelo, ruta_pkl, compress=3)

    # --------------------------------------------------------
    # GUARDAR PREDICCIONES
    # --------------------------------------------------------

    ruta_predicciones, _ = guardar_predicciones(
        nombre_tecnico,
        validacion,
        pred_validacion,
        test,
        pred_test,
    )

    # --------------------------------------------------------
    # REGISTRAR EN LA COMPARACIÓN
    # --------------------------------------------------------

    registro = {
        "modelo": nombre_legible,
        "nombre_tecnico": nombre_tecnico,

        "mae_validacion": metricas_validacion["mae"],
        "rmse_validacion": metricas_validacion["rmse"],
        "mape_validacion": metricas_validacion["mape"],
        "r2_validacion": metricas_validacion["r2"],

        "mae_test": metricas_test["mae"],
        "rmse_test": metricas_test["rmse"],
        "mape_test": metricas_test["mape"],
        "r2_test": metricas_test["r2"],

        "entrenado": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),
    }

    registrar_resultado(registro)

    # --------------------------------------------------------
    # GUARDAR METADATOS
    # --------------------------------------------------------

    guardar_metadatos(
        nombre_tecnico,
        {
            "nombre": nombre_legible,
            "variables": VARIABLES,
            "variable_objetivo": VARIABLE_OBJETIVO,
            "registros_entrenamiento": int(len(train)),
            "periodo_entrenamiento": [
                str(train["fecha"].min().date()),
                str(train["fecha"].max().date()),
            ],
            "metricas_validacion": metricas_validacion,
            "metricas_test": metricas_test,
            "intervalo_80": {
                "inferior": intervalo_inferior,
                "superior": intervalo_superior,
            },
            "intervalo_80_escalado": {
                "inferior": list(escalado_inferior),
                "superior": list(escalado_superior),
            },
            "entrenado": registro["entrenado"],
        },
    )

    print()
    print("Modelo guardado en:")
    print(f"  {ruta_pkl}")
    print("Predicciones guardadas en:")
    print(f"  {ruta_predicciones}")

    return modelo, {
        "validacion": metricas_validacion,
        "test": metricas_test,
    }


# ============================================================
# CARGA DE MODELOS ENTRENADOS
# ============================================================

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo previamente entrenado.
    """

    ruta = ruta_modelo(nombre_modelo)

    if not ruta.exists():

        raise FileNotFoundError(
            f"No existe el modelo entrenado:\n{ruta}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py"
        )

    return joblib.load(ruta)
