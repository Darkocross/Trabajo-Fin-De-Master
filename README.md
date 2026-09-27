# Predicción de demanda en hostelería

**Trabajo Fin de Máster** · Análisis de Datos e Inteligencia Artificial
Juan Francisco Morales Olivo

---

## Qué hace este proyecto

Estima **cuántos clientes va a tener un restaurante un día concreto**, a partir de la previsión meteorológica y del calendario, y traduce esa estimación en una **recomendación de plantilla**.

El problema es real y todo el mundo que ha trabajado en hostelería lo conoce: se decide el personal del fin de semana por intuición, y se falla en las dos direcciones. Unos días sobra gente sin hacer nada y otros el local se desborda. Ninguna de las dos cuesta poco.

El sistema no pretende sustituir al encargado. Le da una cifra de partida, un rango de incertidumbre y una explicación de por qué sale ese número, para que decida con algo más que la sensación de la semana pasada.

**Caso de estudio:** un restaurante en Arganda del Rey (Comunidad de Madrid), con datos meteorológicos reales de AEMET desde 2022.

---

## Resultado

| Modelo | Error medio en validación | Error medio en test | R² test |
|---|---|---|---|
| **Regresión lineal** ← desplegado | 16,19 | **17,01** | **0,736** |
| Hist Gradient Boosting | **16,02** | 17,67 | 0,723 |
| Random Forest | 16,73 | 17,31 | 0,692 |
| Baseline (media por día de la semana) | 20,58 | 20,61 | 0,611 |

*Error medio absoluto, en número de clientes. Menos es mejor.*

**Y lo que de verdad importa: la calidad de la decisión.**

| | Plantilla adecuada | Falta de personal |
|---|---|---|
| Intuición | 82,0 % de los días | 17 días |
| **Con el sistema** | **93,4 %** | **6 días** |

Sobre los 167 días de 2026. El ahorro anual estimado ronda los **3.704 €**, y la conclusión se mantiene en los doce escenarios de sensibilidad probados.

Cuatro conclusiones que merece la pena destacar:

**Los tres modelos empatan, y eso es el hallazgo.** La diferencia entre la regresión lineal y el boosting es de 0,17 clientes en validación, con un intervalo de confianza del 95 % que va de −1,26 a +1,60: estadísticamente indistinguible de cero. El criterio declarado de antemano, aplicado con rigor, **no desempata**. Se despliega la regresión lineal por los motivos secundarios que también estaban declarados: es la única que se puede explicar en una reunión, y gana en test (17,01 frente a 17,67) y en R².

**La mejora sobre el baseline es del 21 % en validación y del 17 % en test.** Es real, pero moderada. El día de la semana explica la mayor parte de la demanda y el baseline ya lo usa; lo que aportan los modelos es la estacionalidad, los festivos y la meteorología.

**El modelo tiene un sesgo conocido, y corregirlo depende del horizonte.** Se queda corto una media de 11 clientes porque el negocio crece un 4 % anual y ninguna de sus variables le dice que ha pasado el tiempo. Se probaron cuatro correcciones: añadir una tendencia extrapolable **empeora mucho** la decisión (de 93,4 % a 83,8 %), porque cambia quedarse corto por pasarse. Reentrenar con una ventana móvil sí ayuda cuando el modelo lleva tiempo sin actualizarse. Está medido en `src/analysis/correccion_tendencia.py` y vigilado por `src/analysis/monitorizacion.py`.

**Y ha aprendido los mecanismos correctos.** Como los datos de actividad están generados, se puede comprobar si el modelo recupera los efectos que realmente los producen: la correlación entre el efecto real y el que atribuye es de **0,971**. La misma prueba destapa su punto ciego: por encima de 33 °C predice en la dirección equivocada.

---

## Cómo ejecutarlo

```bash
# 1. Entorno
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 2. Pipeline completo (unos 30 segundos)
python pipeline.py

# 3. Aplicación
streamlit run app/app.py
```

El pipeline funciona **sin conexión**: los datos descargados de AEMET están en el repositorio, así que todo es reproducible tal cual.

Para volver a descargar de AEMET hace falta una API key gratuita:

```bash
cp .env.example .env
# y escribir la clave dentro

python pipeline.py --todo
```

La aplicación muestra dos cosas: la previsión de un día concreto, con la meteorología que introduzcas, y **la previsión de los próximos días con la predicción real de AEMET**, con su intervalo, la media histórica de cada día y la plantilla recomendada. Para que la semana aparezca hace falta haber descargado la previsión:

```bash
python -m src.extraction.descargar_prediccion
```

### Otras opciones del pipeline

```bash
python pipeline.py --listar            # ver las etapas
python pipeline.py --hasta gold        # solo hasta la capa gold
python pipeline.py --desde baseline    # solo el modelado en adelante
python pipeline.py --solo importancia  # una etapa suelta
```

---

## Estructura

```
.
├── pipeline.py                  Orquestador de todo el proyecto
├── requirements.txt
├── .env.example                 Plantilla de configuración
│
├── app/
│   └── app.py                   Aplicación Streamlit (MVP)
│
├── data/
│   ├── raw/                     Datos de origen
│   │   ├── meteorologia/        AEMET: histórico y previsión
│   │   └── hosteleria/          Actividad diaria del restaurante
│   ├── processed/               Calendario, conjuntos y resultados
│   │   ├── calendario/
│   │   ├── modelado/            train / validation / test
│   │   ├── resultados/          Métricas, predicciones, importancias
│   │   └── graficos/
│   ├── gold/                    Dataset analítico final
│   └── modelos/                 Modelos entrenados (.pkl) y metadatos
│
├── docs/
│   ├── entregas/                Entregas 1 a 5 del máster
│   ├── presentacion/            Defensa (.pptx generado por código)
│   └── assets/                  Mockup del frontal
│
├── notebooks/                   Análisis exploratorio (ejecutados)
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_analisis_demanda.ipynb
│   ├── 03_exploracion_meteorologia.ipynb
│   └── 04_evaluacion_modelo.ipynb
│
├── src/
│   ├── config.py                Rutas y parámetros (única fuente)
│   ├── extraction/              Descarga y archivo de AEMET
│   ├── transformation/          Calendario, capa gold, particiones
│   ├── models/                  Variables, modelos y predicción
│   ├── analysis/                Decisión, robustez, variables, tendencia,
│   │                            monitorización, validación
│   └── visualization/           Gráficos, análisis de error, mockup,
│                                previsión semanal
│
└── tests/                       120 tests con pytest
```

---

## Las tres capas de datos

```
      raw                    processed                 gold
       │                          │                      │
  AEMET 3182Y  ──┐           calendario.csv              │
  (real)         │                │                      │
                 ├──────────► unión por fecha ──► demanda_restaurante.csv
  actividad    ──┘           filtrado de cierres         │
  (sintética)                                     1.129 días de actividad
                                                        │
                                              train / validation / test
                                              2022-24 / 2025 / 2026
```

**Capa raw.** Lo que llega de fuera, sin tocar. La meteorología es real, descargada de la estación 3182Y de AEMET en Arganda del Rey.

**Capa processed.** El calendario generado por reglas y los conjuntos de modelado. El calendario es la única fuente de verdad sobre festivos, vacaciones y días de cierre.

**Capa gold.** Una fila por día de actividad, con todo lo necesario para modelar. Los días de cierre se excluyen: un día cerrado no es un día de cero clientes, es un día sin observación.

---

## Sobre los datos

**La meteorología es real.** Estación 3182Y de AEMET (Arganda del Rey, Madrid), desde el 1 de enero de 2022. Se descarga a través de la API pública de AEMET OpenData.

Tiene huecos, como todo dato real, y el proyecto los trata con cuidado: la capa raw conserva los códigos que envía AEMET (`Ip` es precipitación inapreciable, una medición, no un hueco), el descargador vuelve a pedir los días que faltan porque AEMET publica con retraso, y lo que finalmente se imputa queda marcado por tipo en `temp_interpolada` y `prec_rellenada`. Hay un episodio que conviene conocer: el pluviómetro de la estación estuvo sin dar dato del 19/10/2022 al 20/11/2022, 33 días con las temperaturas correctas. Está documentado en la memoria.

**Los datos de actividad son reales** y corresponden a registros proporcionados por un restaurante.

Para obtenerlos se utiliza el script (`src/transformation/generar_hosteleria.py`), que realiza la descarga de los datos a través de la API proporcionada por el propio restaurante.

Por motivos de confidencialidad y debido al NDA (acuerdo de confidencialidad) existente con el establecimiento, el script utilizado para acceder a dicha API no puede presentarse ni publicarse como parte del proyecto. Por este mismo motivo, tampoco se incluyen credenciales, endpoints privados ni información que permita identificar o acceder directamente a los sistemas del restaurante.

---

## Decisiones que conviene conocer

Las tres que más condicionaron el proyecto:

**`tmed` no es una variable independiente.** AEMET no mide la temperatura media: la calcula como `(tmax + tmin) / 2`. Las tres columnas solo tienen dos grados de libertad, y meter las tres en una regresión daba un VIF por encima de 45.000 y coeficientes que se anulaban entre sí (−104 para la media, +58 para la máxima). El modelo usa `tmed` y `amplitud_termica`, que contienen la misma información sin la colinealidad. El VIF baja a 1,28.

**`n_empleados` no es una variable de entrada.** Se decide a partir del día de la semana, así que no aporta información nueva. Y sobre todo: la dirección útil para el negocio es la contraria. Primero se estiman los clientes y a partir de ahí se recomienda la plantilla. Usarla como entrada haría el producto circular.

**La separación es temporal, no aleatoria.** Entrenar con 2022-2024, validar con 2025 y probar con 2026. Una separación aleatoria mezclaría días de 2026 en el entrenamiento y daría un error mucho mejor del real. En un problema temporal eso no es una simplificación: es un error.

**El calendario contiene hechos, no hipótesis.** Una columna solo entra en `calendario.csv` si se puede comprobar contra el BOE, el convenio o el almanaque: es festivo, es víspera, es puente, el local cierra. "Mayo es un mes de alta demanda" no pasa esa prueba, y de hecho las columnas `es_mayo`/`es_junio`/`es_julio` que había antes no salían de los datos, sino de los tres meses que amplifica el generador sintético. Las interacciones viven ahora en `src/models/features.py`, se construyen para los doce meses y se miden: `python pipeline.py --solo variables` reproduce la comparación.

El razonamiento completo está en `src/models/features.py` y en la Entrega 4.

---

## Tests

```bash
pytest tests/ -v
```

120 tests que cubren el cálculo de la Pascua y los festivos oficiales, la coherencia de la capa gold, las decisiones de `features.py` (que no vuelvan las variables colineales, que no entre ninguna variable de resultado), las métricas y el comportamiento del sistema de predicción.

Algunos son comprobaciones de negocio más que de código: que el sábado prediga más que el miércoles, que la lluvia reduzca la predicción, que la plantilla recomendada nunca deje el local en riesgo de falta de personal, y que el ahorro económico se mantenga en todos los escenarios de sensibilidad.

---

## Documentación

| Documento | Contenido |
|---|---|
| [Memoria del TFM](docs/memoria.md) | Documento completo que integra todo el trabajo |
| [Entrega 1](docs/entregas/01_ideas_producto.md) | Ideas iniciales de proyecto |
| [Entrega 2](docs/entregas/02_datos_necesarios.md) | Selección de la idea y datos necesarios |
| [Entrega 3](docs/entregas/03_modelo_datos.md) | Modelo de datos y capa gold |
| [Entrega 4](docs/entregas/04_analisis_modelado.md) | Análisis, modelado y resultados |
| [Entrega 5](docs/entregas/05_diseno_frontal.md) | Diseño del frontal y experiencia de usuario |
| [Presentación de defensa](docs/presentacion/) | 18 diapositivas con notas, generadas por código |

---

## Fuentes

- **AEMET OpenData** — https://opendata.aemet.es
  Histórico de valores climatológicos diarios (estación 3182Y) y predicción municipal a 7 días (código INE 28014).
- **Calendario laboral de la Comunidad de Madrid** y fiestas locales de Arganda del Rey.

---

## Licencia y uso

Proyecto académico. Los datos meteorológicos son públicos y pertenecen a AEMET. Los datos de actividad son reales pero ocultan la identidad del restaurante por el NDA.
