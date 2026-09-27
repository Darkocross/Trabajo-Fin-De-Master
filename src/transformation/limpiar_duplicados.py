# src/transformation/limpiar_duplicados.py

# ============================================================
# LIMPIEZA DE DUPLICADOS
# ============================================================

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

# Permite ejecutar el script directamente desde:
# src/transformation/limpiar_duplicados.py

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config import DATA_DIR


# ============================================================
# FUNCIONES
# ============================================================

def limpiar_archivo(archivo):
    """
    Elimina registros duplicados por fecha de un archivo CSV.

    Si el archivo no contiene la columna 'fecha', se omite.

    Devuelve:
        True  -> si se ha procesado correctamente
        False -> si ha ocurrido un error
    """

    print()
    print(f"Procesando: {archivo}")

    try:

        # ====================================================
        # LEER CSV
        # ====================================================

        df = pd.read_csv(
            archivo
        )

        # ====================================================
        # COMPROBAR COLUMNA FECHA
        # ====================================================

        if "fecha" not in df.columns:

            print(
                "  -> No tiene columna 'fecha'. "
                "Se omite."
            )

            return True

        # ====================================================
        # FILAS ANTES
        # ====================================================

        filas_antes = len(df)

        # ====================================================
        # DUPLICADOS
        # ====================================================

        duplicados = (
            df["fecha"]
            .duplicated()
            .sum()
        )

        if duplicados == 0:

            print(
                "  -> No hay fechas duplicadas."
            )

            return True

        # ====================================================
        # ELIMINAR DUPLICADOS
        # ====================================================

        df = df.drop_duplicates(
            subset=["fecha"],
            keep="first",
        )

        # ====================================================
        # ORDENAR
        # ====================================================

        df = (
            df
            .sort_values("fecha")
            .reset_index(drop=True)
        )

        # ====================================================
        # GUARDAR
        # ====================================================

        df.to_csv(
            archivo,
            index=False,
            encoding="utf-8",
        )

        # ====================================================
        # INFORMACIÓN
        # ====================================================

        filas_despues = len(df)

        print(
            f"  -> Filas antes: "
            f"{filas_antes}"
        )

        print(
            f"  -> Duplicados eliminados: "
            f"{duplicados}"
        )

        print(
            f"  -> Filas después: "
            f"{filas_despues}"
        )

        return True

    except Exception as e:

        print(
            f"  -> ERROR: {e}"
        )

        return False


def obtener_archivos_csv():
    """
    Busca todos los archivos CSV dentro de data/
    y sus subdirectorios.
    """

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"No existe el directorio de datos: "
            f"{DATA_DIR}"
        )

    return sorted(
        DATA_DIR.rglob("*.csv")
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LIMPIEZA DE DUPLICADOS")
    print("=" * 60)

    print()
    print(f"Directorio de datos: {DATA_DIR}")

    # ========================================================
    # BUSCAR ARCHIVOS
    # ========================================================

    archivos = obtener_archivos_csv()

    print()
    print(
        f"Archivos CSV encontrados: "
        f"{len(archivos)}"
    )

    if not archivos:

        print()
        print(
            "No se han encontrado archivos CSV."
        )

        return

    # ========================================================
    # PROCESAR
    # ========================================================

    procesados = 0
    errores = 0

    for archivo in archivos:

        resultado = limpiar_archivo(
            archivo
        )

        if resultado:

            procesados += 1

        else:

            errores += 1

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print(
        f"Archivos encontrados: {len(archivos)}"
    )

    print(
        f"Archivos procesados: {procesados}"
    )

    print(
        f"Errores: {errores}"
    )

    print()
    print("Proceso terminado.")

    # Si ha habido errores, hacemos que el pipeline
    # sepa que el proceso no ha terminado correctamente.
    if errores > 0:

        raise RuntimeError(
            f"Se produjeron errores en "
            f"{errores} archivo(s)."
        )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()