# src/analysis/robustez_prevision.py

# ============================================================
# ROBUSTEZ FRENTE AL ERROR DE LA PREVISIÓN
# ============================================================
#
# Hay un hueco en toda la evaluación anterior, y conviene
# ponerlo encima de la mesa.
#
# El modelo se entrena y se evalúa con la meteorología
# OBSERVADA: la que realmente hizo. Pero cuando el encargado
# usa el sistema, el día todavía no ha ocurrido. Lo único que
# tiene es la PREVISIÓN, que se equivoca.
#
# Es decir: el MAE que publica el proyecto corresponde a un
# escenario imposible, en el que se conoce el tiempo con
# certeza. El error real en producción es necesariamente mayor.
#
# ------------------------------------------------------------
# POR QUÉ ES UN TEST DE ESTRÉS Y NO UNA MEDICIÓN
# ------------------------------------------------------------
#
# Lo ideal sería comparar, para cada día, la previsión que dio
# AEMET con lo que luego se observó. No se puede: AEMET publica
# la predicción a 7 días, pero no un archivo histórico de las
# predicciones pasadas. Para construir ese conjunto hay que ir
# guardando la previsión cada día, y el proyecto no lleva
# funcionando el tiempo suficiente.
#
# Así que, en lugar de inventar una cifra de error de AEMET, se
# hace lo contrario: se DEGRADA la meteorología a propósito, en
# un rango amplio, y se mide cuánto empeora el sistema en cada
# nivel. La conclusión no depende entonces de acertar cuál es
# el error real de AEMET: se lee en la curva.
#
# `src/extraction/archivar_prediccion.py` guarda cada día la
# previsión, para poder hacer la medición de verdad más
# adelante.
#
# ------------------------------------------------------------
# CÓMO SE DEGRADA
# ------------------------------------------------------------
#
# TEMPERATURA. Se suma ruido normal a la máxima y a la mínima.
# El error de una previsión no es independiente entre ambas: si
# el modelo meteorológico se equivoca con la masa de aire, falla
# en las dos a la vez. Se reparte el ruido en una componente
# común al día y otra propia de cada variable.
#
# LLUVIA. El fallo dominante no es equivocarse en los
# milímetros, sino en el sí o el no: prever lluvia que no cae, o
# no verla venir. Se modela con una probabilidad de confundir
# día seco y día lluvioso, más ruido multiplicativo en la
# cantidad.
#
# ============================================================

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    PROCESSED_RESULTADOS_DIR,
    SEMILLA,
    TEST,
    ruta_grafico,
)

from src.models.entrenamiento import calcular_metricas
from src.models.features import obtener_x, preparar_dataset
from src.models.prediccion import cargar_modelo_produccion

from src.analysis.evaluacion_decision import (
    evaluar_metodo,
)


SALIDA = PROCESSED_RESULTADOS_DIR / "robustez_prevision.csv"


# ============================================================
# ESCENARIOS DE HORIZONTE
# ============================================================
#
# Estos valores son SUPUESTOS razonables sobre cómo se degrada
# una previsión al alejarse el horizonte, no medidas de la
# pericia real de AEMET. Están aquí para poder señalar puntos
# concretos sobre la curva; la curva completa es lo que
# sostiene la conclusión.

HORIZONTES = [
    {
        "nombre": "Observado",
        "dias": 0,
        "sigma_temperatura": 0.0,
        "prob_fallo_lluvia": 0.00,
    },
    {
        "nombre": "1 día",
        "dias": 1,
        "sigma_temperatura": 1.0,
        "prob_fallo_lluvia": 0.08,
    },
    {
        "nombre": "3 días",
        "dias": 3,
        "sigma_temperatura": 1.8,
        "prob_fallo_lluvia": 0.18,
    },
    {
        "nombre": "5 días",
        "dias": 5,
        "sigma_temperatura": 2.4,
        "prob_fallo_lluvia": 0.27,
    },
    {
        "nombre": "7 días",
        "dias": 7,
        "sigma_temperatura": 3.0,
        "prob_fallo_lluvia": 0.35,
    },
]


# Repeticiones de la simulación. Cada una usa un sorteo
# distinto del error, y se promedia.
REPETICIONES = 40


# ============================================================
# DEGRADACIÓN DE LA METEOROLOGÍA
# ============================================================

def degradar_meteorologia(
    datos,
    sigma_temperatura,
    prob_fallo_lluvia,
    rng,
):
    """
    Devuelve una copia de los datos con la meteorología
    sustituida por una "previsión" con error.
    """

    datos = datos.copy()

    n = len(datos)

    if sigma_temperatura > 0:

        # --- Componente común del día ---
        #
        # Recoge el fallo del modelo meteorológico con la masa
        # de aire, que afecta a la máxima y a la mínima a la
        # vez.

        comun = rng.normal(0, sigma_temperatura * 0.75, n)

        propio_max = rng.normal(
            0, sigma_temperatura * 0.66, n
        )

        propio_min = rng.normal(
            0, sigma_temperatura * 0.66, n
        )

        tmax = datos["tmax"].to_numpy() + comun + propio_max
        tmin = datos["tmin"].to_numpy() + comun + propio_min

        # La mínima no puede superar a la máxima.
        tmin = np.minimum(tmin, tmax - 0.1)

        datos["tmax"] = tmax
        datos["tmin"] = tmin
        datos["tmed"] = (tmax + tmin) / 2

    if prob_fallo_lluvia > 0:

        prec = datos["prec"].to_numpy(dtype=float)

        llovio = prec > 0.5

        falla = rng.random(n) < prob_fallo_lluvia

        nueva = prec.copy()

        # --- No ver venir la lluvia ---

        nueva[llovio & falla] = 0.0

        # --- Prever lluvia que no cae ---
        #
        # Se le asigna una cantidad típica de día lluvioso.

        lluvia_tipica = (
            prec[llovio].mean() if llovio.any() else 5.0
        )

        inventada = (~llovio) & falla

        nueva[inventada] = rng.gamma(
            2.0,
            lluvia_tipica / 2.0,
            inventada.sum(),
        )

        # --- Ruido en la cantidad de los aciertos ---

        acierta_lluvia = llovio & (~falla)

        nueva[acierta_lluvia] *= rng.lognormal(
            0,
            0.55,
            acierta_lluvia.sum(),
        )

        datos["prec"] = np.clip(nueva, 0, None)

    return datos


# ============================================================
# SIMULACIÓN
# ============================================================

def simular_horizonte(
    modelo,
    test,
    sigma_temperatura,
    prob_fallo_lluvia,
    repeticiones=REPETICIONES,
):
    """
    Repite la predicción con meteorología degradada y devuelve
    el error medio y el impacto sobre la decisión de plantilla.
    """

    reales = test["n_clientes"].to_numpy(dtype=float)

    maes = []
    rmses = []
    dias_cortos = []
    dias_correctos = []

    for repeticion in range(repeticiones):

        rng = np.random.default_rng(
            SEMILLA + repeticion * 1000
        )

        degradado = degradar_meteorologia(
            test,
            sigma_temperatura,
            prob_fallo_lluvia,
            rng,
        )

        # Se vuelven a derivar las variables (amplitud
        # térmica, lluvia en escala logarítmica, interacción)
        # a partir de la meteorología degradada.
        degradado = preparar_dataset(degradado)

        predicciones = np.clip(
            modelo.predict(obtener_x(degradado)),
            0,
            None,
        )

        metricas = calcular_metricas(reales, predicciones)

        maes.append(metricas["mae"])
        rmses.append(metricas["rmse"])

        decision = evaluar_metodo(reales, predicciones)

        dias_cortos.append(
            int(decision["infradotado"].sum())
        )

        dias_correctos.append(
            int(decision["correcto"].sum())
        )

    return {
        "mae": float(np.mean(maes)),
        "mae_desviacion": float(np.std(maes)),
        "rmse": float(np.mean(rmses)),
        "dias_falta_personal": float(np.mean(dias_cortos)),
        "dias_correctos": float(np.mean(dias_correctos)),
    }


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_robustez(barrido, horizontes, ruta):
    """
    Curva de degradación y puntos de los horizontes.
    """

    figura, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Curva continua ---

    ejes[0].plot(
        barrido["sigma_temperatura"],
        barrido["mae"],
        marker="o",
        linewidth=2,
        color="#1f4e79",
    )

    ejes[0].fill_between(
        barrido["sigma_temperatura"],
        barrido["mae"] - barrido["mae_desviacion"],
        barrido["mae"] + barrido["mae_desviacion"],
        color="#1f4e79",
        alpha=0.15,
    )

    base = barrido["mae"].iloc[0]

    ejes[0].axhline(
        base,
        color="#27ae60",
        linestyle="--",
        linewidth=1.4,
        label=f"con meteorología observada: {base:.1f}",
    )

    ejes[0].set_xlabel(
        "Error de la previsión de temperatura (°C)"
    )

    ejes[0].set_ylabel("Error medio absoluto (clientes)")

    ejes[0].set_title(
        "Cuánto empeora el modelo al degradar la previsión"
    )

    ejes[0].legend(fontsize=9)

    # --- Por horizonte ---

    tabla = pd.DataFrame(horizontes)

    posiciones = np.arange(len(tabla))

    colores = [
        "#27ae60" if fila == 0 else "#1f4e79"
        for fila in tabla["dias"]
    ]

    ejes[1].bar(
        posiciones,
        tabla["mae"],
        color=colores,
        edgecolor="white",
    )

    for posicion, (mae, correctos) in enumerate(
        zip(tabla["mae"], tabla["dias_correctos"])
    ):

        ejes[1].text(
            posicion,
            mae + 0.4,
            f"{mae:.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ejes[1].set_xticks(posiciones)
    ejes[1].set_xticklabels(tabla["nombre"])

    ejes[1].set_ylabel("Error medio absoluto (clientes)")

    ejes[1].set_title(
        "Error esperado según la antelación\n"
        "(escenarios supuestos, no medidos)"
    )

    for eje in ejes:
        eje.grid(axis="y", alpha=0.3)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)

    figura.tight_layout()

    ruta.parent.mkdir(parents=True, exist_ok=True)

    figura.savefig(ruta, dpi=150)

    plt.close(figura)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ROBUSTEZ FRENTE AL ERROR DE LA PREVISIÓN")
    print("=" * 60)

    modelo, metadatos = cargar_modelo_produccion()

    test = pd.read_csv(TEST, parse_dates=["fecha"])

    test = preparar_dataset(test)

    print()
    print(f"Días evaluados: {len(test)}")
    print(f"Repeticiones por escenario: {REPETICIONES}")

    print()
    print(
        "El error de la previsión NO está medido: se simula "
        "degradando"
    )
    print(
        "la meteorología observada en un rango amplio. Ver la "
        "cabecera"
    )
    print("del módulo.")

    # --------------------------------------------------------
    # BARRIDO CONTINUO
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DEGRADACIÓN PROGRESIVA")
    print("=" * 60)
    print()

    filas_barrido = []

    for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:

        # La probabilidad de fallar con la lluvia se escala
        # junto con el error de temperatura.
        prob = min(0.40, sigma * 0.11)

        resultado = simular_horizonte(
            modelo,
            test,
            sigma,
            prob,
        )

        filas_barrido.append({
            "sigma_temperatura": sigma,
            "prob_fallo_lluvia": round(prob, 3),
            **{
                clave: round(valor, 2)
                for clave, valor in resultado.items()
            },
        })

        print(
            f"  ±{sigma:.1f} °C, "
            f"{100 * prob:>4.1f} % de fallo con la lluvia   "
            f"->  MAE {resultado['mae']:5.2f}   "
            f"días cortos de personal: "
            f"{resultado['dias_falta_personal']:.1f}"
        )

    barrido = pd.DataFrame(filas_barrido)

    # --------------------------------------------------------
    # HORIZONTES
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ERROR ESPERADO SEGÚN LA ANTELACIÓN")
    print("=" * 60)

    filas_horizonte = []

    for horizonte in HORIZONTES:

        resultado = simular_horizonte(
            modelo,
            test,
            horizonte["sigma_temperatura"],
            horizonte["prob_fallo_lluvia"],
        )

        filas_horizonte.append({
            **horizonte,
            **{
                clave: round(valor, 2)
                for clave, valor in resultado.items()
            },
        })

    tabla = pd.DataFrame(filas_horizonte)

    base = tabla.iloc[0]

    print()
    print(
        f"{'antelación':<12} {'MAE':>7} {'vs observado':>14} "
        f"{'días correctos':>16} {'días cortos':>13}"
    )

    print("-" * 68)

    for _, fila in tabla.iterrows():

        incremento = (
            100 * (fila["mae"] / base["mae"] - 1)
            if base["mae"]
            else 0
        )

        print(
            f"{fila['nombre']:<12} "
            f"{fila['mae']:>7.2f} "
            f"{incremento:>12.1f} % "
            f"{fila['dias_correctos']:>14.1f} "
            f"{fila['dias_falta_personal']:>13.1f}"
        )

    # --------------------------------------------------------
    # LECTURA
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("QUÉ SIGNIFICA")
    print("=" * 60)

    un_dia = tabla[tabla["nombre"] == "1 día"].iloc[0]
    siete = tabla[tabla["nombre"] == "7 días"].iloc[0]

    print()

    print(
        f"  Con meteorología observada el MAE es "
        f"{base['mae']:.1f} clientes."
    )

    print(
        f"  Prediciendo con un día de antelación sube a "
        f"{un_dia['mae']:.1f} "
        f"({100 * (un_dia['mae'] / base['mae'] - 1):+.0f} %)."
    )

    print(
        f"  A siete días sube a {siete['mae']:.1f} "
        f"({100 * (siete['mae'] / base['mae'] - 1):+.0f} %)."
    )

    print()

    perdida_dias = (
        base["dias_correctos"] - siete["dias_correctos"]
    )

    print(
        f"  En términos de decisión, pasar de observación a "
        f"siete días"
    )

    print(
        f"  cuesta {perdida_dias:.0f} días de plantilla "
        f"correcta de {len(test)}."
    )

    print()

    if un_dia["mae"] / base["mae"] < 1.15:

        print(
            "  La conclusión operativa es clara: el sistema "
            "aguanta bien la"
        )

        print(
            "  previsión a corto plazo. Planificar con uno o "
            "dos días de"
        )

        print(
            "  antelación apenas cuesta precisión; hacerlo a "
            "una semana, sí."
        )

    print()

    print(
        "  Encaja con el uso real: la plantilla del fin de "
        "semana se cierra"
    )

    print(
        "  el jueves o el viernes, no el lunes anterior."
    )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    PROCESSED_RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    barrido.to_csv(
        SALIDA,
        index=False,
        encoding="utf-8",
    )

    tabla.to_csv(
        PROCESSED_RESULTADOS_DIR / "robustez_horizontes.csv",
        index=False,
        encoding="utf-8",
    )

    grafico_robustez(
        barrido,
        filas_horizonte,
        ruta_grafico("robustez_prevision"),
    )

    print()
    print(f"Guardado en: {SALIDA}")

    print()
    print("=" * 60)
    print("ANÁLISIS DE ROBUSTEZ COMPLETADO")
    print("=" * 60)

    return barrido, tabla


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
