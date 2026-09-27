# src/transformation/validar_gold.py

# ============================================================
# VALIDACIÓN DE LA CAPA GOLD
# ============================================================
#
# Comprueba que la capa gold cumple todo lo que el resto del
# proyecto da por hecho.
#
# Es una red de seguridad: si mañana se cambia el generador de
# datos, el calendario o la descarga de AEMET y algo se rompe,
# este script lo detecta AQUÍ, antes de entrenar ningún modelo
# y antes de que aparezca como un resultado raro en la memoria
# del TFM.
#
# Cada comprobación devuelve si ha pasado o no, y al final se
# resume. Si alguna crítica falla, el pipeline se detiene.
#
# ============================================================

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import (
    AÑO_FIN,
    AÑO_INICIO,
    GOLD_DATASET,
    MES_CIERRE_VACACIONES,
)


# ============================================================
# COLUMNAS OBLIGATORIAS
# ============================================================

COLUMNAS_OBLIGATORIAS = [
    "fecha",
    "año",
    "mes",
    "dia_semana",
    "nombre_dia",
    "fin_de_semana",
    "es_festivo",
    "es_vacaciones",
    "tipo_vacaciones",
    "es_vispera_festivo",
    "es_puente",
    "meteo_imputada",
    "tmed",
    "tmax",
    "tmin",
    "prec",
    "n_empleados",
    "n_clientes",
    "nota_faena",
]


# ============================================================
# REGISTRO DE RESULTADOS
# ============================================================

class Resultados:
    """
    Acumula el resultado de cada comprobación.
    """

    def __init__(self):

        self.comprobaciones = []

    def comprobar(self, nombre, condicion, detalle="", critica=True):

        self.comprobaciones.append({
            "nombre": nombre,
            "correcta": bool(condicion),
            "detalle": detalle,
            "critica": critica,
        })

        simbolo = "OK  " if condicion else "FALLO"

        linea = f"  [{simbolo}] {nombre}"

        if detalle:
            linea += f"  ->  {detalle}"

        print(linea)

        return bool(condicion)

    @property
    def fallos_criticos(self):

        return [
            comprobacion
            for comprobacion in self.comprobaciones
            if not comprobacion["correcta"]
            and comprobacion["critica"]
        ]

    @property
    def avisos(self):

        return [
            comprobacion
            for comprobacion in self.comprobaciones
            if not comprobacion["correcta"]
            and not comprobacion["critica"]
        ]


# ============================================================
# CARGA
# ============================================================

def cargar_gold():

    if not GOLD_DATASET.exists():

        raise FileNotFoundError(
            f"No existe la capa gold: {GOLD_DATASET}\n\n"
            "Ejecuta antes:\n"
            "    python pipeline.py --hasta gold"
        )

    return pd.read_csv(
        GOLD_DATASET,
        parse_dates=["fecha"],
    )


# ============================================================
# COMPROBACIONES
# ============================================================

def validar_estructura(gold, resultados):

    print()
    print("ESTRUCTURA")
    print("-" * 60)

    faltantes = [
        columna
        for columna in COLUMNAS_OBLIGATORIAS
        if columna not in gold.columns
    ]

    resultados.comprobar(
        "Todas las columnas obligatorias están presentes",
        not faltantes,
        f"faltan: {faltantes}" if faltantes else "",
    )

    resultados.comprobar(
        "El dataset no está vacío",
        len(gold) > 0,
        f"{len(gold)} registros",
    )


def validar_fechas(gold, resultados):

    print()
    print("FECHAS")
    print("-" * 60)

    resultados.comprobar(
        "No hay fechas duplicadas",
        gold["fecha"].duplicated().sum() == 0,
        f"{int(gold['fecha'].duplicated().sum())} duplicadas",
    )

    resultados.comprobar(
        "Las fechas están ordenadas",
        gold["fecha"].is_monotonic_increasing,
    )

    resultados.comprobar(
        "El periodo está dentro del rango del proyecto",
        (
            gold["fecha"].dt.year.min() >= AÑO_INICIO
            and gold["fecha"].dt.year.max() <= AÑO_FIN
        ),
        f"{gold['fecha'].min().date()} -> "
        f"{gold['fecha'].max().date()}",
    )

    resultados.comprobar(
        "No hay registros en el futuro",
        gold["fecha"].max() <= pd.Timestamp.today().normalize(),
        f"último: {gold['fecha'].max().date()}",
    )


def validar_coherencia_calendario(gold, resultados):

    print()
    print("COHERENCIA DEL CALENDARIO")
    print("-" * 60)

    resultados.comprobar(
        "`dia_semana` coincide con la fecha",
        (gold["dia_semana"] == gold["fecha"].dt.weekday).all(),
    )

    resultados.comprobar(
        "`mes` coincide con la fecha",
        (gold["mes"] == gold["fecha"].dt.month).all(),
    )

    resultados.comprobar(
        "`fin_de_semana` coincide con `dia_semana`",
        (
            gold["fin_de_semana"]
            == (gold["dia_semana"] >= 5).astype(int)
        ).all(),
    )

    # Un puente es, por definición, un día laborable: no puede
    # ser festivo ni caer en fin de semana.

    puentes = gold["es_puente"] == 1

    comprobaciones_puente = {
        "Ningún puente es festivo": (
            (gold.loc[puentes, "es_festivo"] == 0).all()
        ),
        "Ningún puente cae en fin de semana": (
            (gold.loc[puentes, "fin_de_semana"] == 0).all()
        ),
    }

    for descripcion, condicion in comprobaciones_puente.items():

        resultados.comprobar(
            descripcion,
            bool(condicion),
        )

    resultados.comprobar(
        "No hay actividad en el mes de cierre por vacaciones",
        (gold["mes"] != MES_CIERRE_VACACIONES).all(),
        f"mes {MES_CIERRE_VACACIONES}",
    )


def validar_valores(gold, resultados):

    print()
    print("VALORES")
    print("-" * 60)

    resultados.comprobar(
        "No hay clientes negativos",
        (gold["n_clientes"] >= 0).all(),
    )

    resultados.comprobar(
        "No hay días abiertos con cero clientes",
        (gold["n_clientes"] > 0).all(),
        f"{int((gold['n_clientes'] == 0).sum())} días",
    )

    resultados.comprobar(
        "Los empleados están en un rango razonable",
        gold["n_empleados"].between(1, 20).all(),
        f"{gold['n_empleados'].min()} - "
        f"{gold['n_empleados'].max()}",
    )

    resultados.comprobar(
        "tmin <= tmed <= tmax",
        (
            (gold["tmin"] <= gold["tmed"])
            & (gold["tmed"] <= gold["tmax"])
        ).all(),
    )

    resultados.comprobar(
        "Las temperaturas son plausibles",
        (
            gold["tmin"].between(-20, 50).all()
            and gold["tmax"].between(-20, 55).all()
        ),
        f"{gold['tmin'].min():.1f} a {gold['tmax'].max():.1f} °C",
    )

    resultados.comprobar(
        "La precipitación no es negativa",
        (gold["prec"] >= 0).all(),
    )

    binarias = [
        "es_festivo",
        "es_vacaciones",
        "fin_de_semana",
        "es_vispera_festivo",
        "es_puente",
        "prec_inapreciable",
        "temp_interpolada",
        "prec_rellenada",
        "meteo_imputada",
    ]

    valores_no_binarios = [
        columna
        for columna in binarias
        if columna in gold.columns
        and not gold[columna].isin([0, 1]).all()
    ]

    resultados.comprobar(
        "Las variables binarias solo valen 0 o 1",
        not valores_no_binarios,
        f"incorrectas: {valores_no_binarios}"
        if valores_no_binarios
        else "",
    )


def validar_nulos(gold, resultados):

    print()
    print("VALORES NULOS")
    print("-" * 60)

    criticas = [
        "fecha",
        "dia_semana",
        "mes",
        "n_clientes",
        "tmed",
        "tmax",
        "tmin",
        "prec",
    ]

    nulos = gold[criticas].isna().sum()

    resultados.comprobar(
        "No hay nulos en las columnas críticas",
        nulos.sum() == 0,
        f"{nulos[nulos > 0].to_dict()}" if nulos.sum() else "",
    )

    if "meteo_imputada" in gold.columns:

        imputados = int(gold["meteo_imputada"].sum())

        resultados.comprobar(
            "Los días con meteorología imputada son pocos",
            imputados < 0.05 * len(gold),
            f"{imputados} de {len(gold)} días "
            f"({100 * imputados / len(gold):.1f} %)",
            critica=False,
        )


def validar_cobertura(gold, resultados):

    print()
    print("COBERTURA TEMPORAL")
    print("-" * 60)

    por_año = gold.groupby("año").size()

    # Se excluye el último año, que puede estar incompleto.
    años_completos = por_año.iloc[:-1]

    resultados.comprobar(
        "Todos los años completos tienen datos suficientes",
        (años_completos >= 200).all() if len(años_completos) else True,
        f"{por_año.to_dict()}",
    )

    dias_semana = gold["dia_semana"].nunique()

    resultados.comprobar(
        "Hay datos de al menos cinco días de la semana",
        dias_semana >= 5,
        f"{dias_semana} días distintos",
    )


# ============================================================
# RESUMEN DESCRIPTIVO
# ============================================================

def mostrar_resumen(gold):

    print()
    print("=" * 60)
    print("RESUMEN DE LA CAPA GOLD")
    print("=" * 60)

    print()
    print(f"Registros: {len(gold)}")

    print(
        f"Periodo:   {gold['fecha'].min().date()} -> "
        f"{gold['fecha'].max().date()}"
    )

    print()
    print("CLIENTES POR DÍA DE LA SEMANA")
    print("-" * 60)

    resumen = (
        gold
        .groupby("nombre_dia")["n_clientes"]
        .agg(["count", "mean", "min", "max"])
        .round(1)
    )

    orden = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    ]

    resumen = resumen.reindex(
        [dia for dia in orden if dia in resumen.index]
    )

    resumen.columns = ["días", "media", "mínimo", "máximo"]

    print(resumen.to_string())

    print()
    print("CLIENTES POR AÑO")
    print("-" * 60)

    print(
        gold
        .groupby("año")["n_clientes"]
        .agg(["count", "mean"])
        .round(1)
        .rename(columns={"count": "días", "mean": "media"})
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VALIDACIÓN DE LA CAPA GOLD")
    print("=" * 60)

    gold = cargar_gold()

    resultados = Resultados()

    validar_estructura(gold, resultados)
    validar_fechas(gold, resultados)
    validar_coherencia_calendario(gold, resultados)
    validar_valores(gold, resultados)
    validar_nulos(gold, resultados)
    validar_cobertura(gold, resultados)

    mostrar_resumen(gold)

    # --------------------------------------------------------
    # VEREDICTO
    # --------------------------------------------------------

    total = len(resultados.comprobaciones)

    correctas = total - len(resultados.fallos_criticos) - len(
        resultados.avisos
    )

    print()
    print("=" * 60)
    print("RESULTADO DE LA VALIDACIÓN")
    print("=" * 60)
    print()

    print(f"Comprobaciones superadas: {correctas} / {total}")

    if resultados.avisos:

        print()
        print("Avisos:")

        for aviso in resultados.avisos:
            print(f"  - {aviso['nombre']}: {aviso['detalle']}")

    if resultados.fallos_criticos:

        print()
        print("Fallos críticos:")

        for fallo in resultados.fallos_criticos:
            print(f"  - {fallo['nombre']}: {fallo['detalle']}")

        raise ValueError(
            f"La capa gold no supera "
            f"{len(resultados.fallos_criticos)} "
            "comprobación(es) crítica(s)."
        )

    print()
    print("La capa gold es válida.")


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    main()
