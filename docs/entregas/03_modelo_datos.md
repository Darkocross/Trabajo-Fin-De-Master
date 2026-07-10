# Entrega 3 - Diseño del modelo de datos y capa gold del proyecto

**Por Juan Francisco Morales Olivo**

## 1. Resumen de la idea y datos del proyecto

El proyecto que se desarrollará durante el curso tiene como objetivo estudiar la relación existente entre las condiciones meteorológicas y la demanda en establecimientos de hostelería. La idea surge de un problema habitual en este sector: la dificultad para anticipar con suficiente precisión la afluencia de clientes y, como consecuencia, planificar adecuadamente los recursos necesarios para cada jornada.

En muchos negocios las decisiones relacionadas con la contratación de personal, la compra de productos o la organización del servicio se basan principalmente en la experiencia de los responsables del establecimiento. Sin embargo, existen numerosos factores externos que pueden influir en el comportamiento de los clientes. Entre ellos, las condiciones meteorológicas juegan un papel importante, especialmente en negocios cuya actividad depende en gran medida del consumo presencial.

La solución propuesta consiste en desarrollar un modelo de análisis de datos que permita identificar la relación entre variables meteorológicas y la demanda observada en distintos establecimientos. A partir de esa información se entrenará un modelo predictivo capaz de estimar el número esperado de clientes o el volumen de ventas para una fecha determinada utilizando la previsión meteorológica como variable de entrada.

Para construir este modelo será necesario combinar información procedente de diferentes fuentes. Por un lado, se utilizarán datos relacionados con la actividad del negocio, como el número de clientes, el número de tickets emitidos o el importe total de ventas registrado cada día. Por otro lado, se recopilarán datos meteorológicos históricos correspondientes a las mismas fechas y ubicaciones, incluyendo variables como la temperatura, la precipitación, la humedad relativa o la velocidad del viento.

La principal fuente de información meteorológica será la Agencia Estatal de Meteorología (AEMET), mediante su servicio OpenData, aunque también se contempla la posibilidad de utilizar Open-Meteo como fuente complementaria si fuese necesario. En cuanto a los datos de actividad del negocio, la intención inicial es trabajar con un conjunto de datos simulado que reproduzca un comportamiento realista, aunque también se estudiará la posibilidad de utilizar datasets públicos relacionados con ventas o afluencia de clientes si se encuentran fuentes adecuadas.

Toda esta información se integrará posteriormente en un único conjunto de datos preparado para realizar el análisis exploratorio, entrenar el modelo predictivo y desarrollar un dashboard que permita visualizar tanto el comportamiento histórico como las predicciones obtenidas.

---

## 2. Tecnología o formato de almacenamiento elegido

El proyecto utilizará distintos formatos de almacenamiento en función de la fase en la que se encuentren los datos. La intención no es incorporar tecnologías complejas, sino utilizar aquellas que mejor se adapten al volumen de información previsto y al flujo de trabajo que se seguirá durante el desarrollo del proyecto.

Aunque inicialmente valoré la posibilidad de utilizar una base de datos relacional como SQLite o PostgreSQL, finalmente considero que no es necesario para este caso de estudio. El volumen de información con el que se trabajará será relativamente reducido y los datos se procesarán principalmente mediante Python y la biblioteca Pandas, por lo que resulta más sencillo trabajar directamente con ficheros.

### Datos originales

Los datos obtenidos de las distintas fuentes se almacenarán inicialmente respetando su formato original siempre que sea posible.

En el caso de la información meteorológica procedente de la API de AEMET, las respuestas se recibirán en formato JSON. Estos ficheros se conservarán sin modificar dentro de la capa de datos originales para disponer siempre de una copia de la información tal y como ha sido proporcionada por la fuente.

Por otro lado, los datos relacionados con la actividad del negocio, ya sean simulados o procedentes de conjuntos de datos abiertos, se almacenarán inicialmente en formato CSV. Este formato resulta sencillo de generar, ampliamente compatible con herramientas de análisis y suficiente para el volumen de información previsto.

### Datos procesados

Una vez realizada la extracción de la información, los datos pasarán por una fase de limpieza y transformación.

En esta etapa se convertirán al formato Parquet. La elección de este formato se debe a varias razones:

- Permite almacenar los datos de forma más eficiente que un fichero CSV.
- Reduce el espacio ocupado en disco.
- Conserva correctamente los tipos de datos.
- Ofrece una lectura y escritura más rápida durante el análisis con Pandas.
- Es un formato ampliamente utilizado en proyectos de análisis de datos.

Además, trabajar con ficheros Parquet facilitará el entrenamiento del modelo predictivo y la generación de visualizaciones sin necesidad de realizar conversiones adicionales en cada ejecución.

### Datos finales

El conjunto de datos definitivo, correspondiente a la capa *gold*, también se almacenará en formato Parquet.

Este dataset será el único utilizado por las fases posteriores del proyecto, incluyendo el análisis exploratorio de datos, el entrenamiento del modelo de predicción y el desarrollo del dashboard final.

De esta forma se garantiza que todas las etapas del proyecto trabajen sobre una única versión consolidada de los datos, evitando inconsistencias entre distintos ficheros o procesos de transformación.

### Justificación de la elección

La combinación de JSON, CSV y Parquet permite mantener una estructura sencilla y fácilmente reproducible.

Los ficheros JSON conservarán los datos originales descargados desde las APIs, los CSV facilitarán el intercambio de información cuando sea necesario y los ficheros Parquet actuarán como formato principal para el procesamiento y análisis de los datos.

Considero que esta solución es suficiente para el alcance del proyecto y evita incorporar herramientas adicionales cuya complejidad no aportaría un beneficio real en un trabajo de estas dimensiones.

---

## 3. Estructura de capas de datos

Con el objetivo de mantener una organización clara y facilitar la trazabilidad de los datos durante todo el desarrollo del proyecto, se utilizará una estructura basada en tres capas: **raw**, **processed** y **gold**.

Esta organización permitirá conservar siempre los datos originales, documentar las transformaciones realizadas y disponer de un único conjunto de datos preparado para el análisis y el entrenamiento del modelo.

La estructura prevista para el directorio de datos será la siguiente:

```text
data/
│
├── raw/
│   ├── meteorologia/
│   ├── ventas/
│   └── festivos/
│
├── processed/
│   ├── meteorologia.parquet
│   ├── ventas.parquet
│   └── festivos.parquet
│
└── gold/
    └── gold_demanda_hosteleria.parquet
```

### Capa raw

La carpeta **raw** contendrá los datos originales obtenidos de las distintas fuentes. En esta capa no se realizarán modificaciones sobre el contenido de los archivos, salvo aquellas estrictamente necesarias para poder almacenarlos correctamente.

El objetivo de esta capa es conservar una copia fiel de la información descargada, de forma que el proceso completo de transformación pueda repetirse en cualquier momento si fuera necesario.

Dentro de esta carpeta se almacenarán, principalmente:

- Respuestas descargadas desde la API de AEMET.
- Datos obtenidos de Open-Meteo, si finalmente se utilizan.
- Datos de ventas o actividad del negocio.
- Información relacionada con calendarios de festivos.

Mantener estos datos sin modificar facilita la reproducibilidad del proyecto y permite comprobar posteriormente que todas las transformaciones realizadas son correctas.

### Capa processed

La capa **processed** contendrá una versión limpia y normalizada de cada una de las fuentes de datos.

En esta fase se realizarán las primeras transformaciones necesarias para preparar la información antes de combinarla. Algunas de las tareas previstas son:

- Conversión de tipos de datos.
- Unificación del formato de fechas.
- Corrección de nombres de columnas.
- Eliminación de registros duplicados.
- Tratamiento inicial de valores nulos.
- Eliminación de registros claramente erróneos.

Cada conjunto de datos continuará almacenándose de forma independiente. Por ejemplo, los datos meteorológicos y los datos de ventas seguirán siendo archivos distintos, ya que todavía no se habrán combinado entre sí.

Trabajar con una capa intermedia facilita la validación de los datos y permite detectar errores antes de construir el dataset definitivo.

### Capa gold

La última capa del proceso será la capa **gold**, que contendrá el conjunto de datos final preparado para ser utilizado durante el resto del proyecto.

En esta fase se combinarán los datos meteorológicos con la información de actividad del negocio mediante las claves comunes definidas durante el proceso de integración.

El resultado será un único dataset limpio, consistente y documentado, sobre el que se desarrollarán todas las fases posteriores del proyecto.

Concretamente, este dataset será utilizado para:

- Realizar el análisis exploratorio de los datos (EDA).
- Entrenar y evaluar el modelo predictivo.
- Generar las visualizaciones del proyecto.
- Alimentar el dashboard final.
- Elaborar las conclusiones del trabajo.

De esta forma, cualquier análisis realizado durante el proyecto utilizará siempre la misma versión de los datos, evitando inconsistencias entre distintas fases del desarrollo.

### Flujo de datos

El flujo previsto de la información puede resumirse mediante el siguiente esquema:

```text
Fuentes de datos
        │
        ▼
   Capa raw
        │
        ▼
Limpieza y transformación
        │
        ▼
 Capa processed
        │
        ▼
Integración de fuentes
        │
        ▼
    Capa gold
        │
        ├── Análisis exploratorio (EDA)
        ├── Modelo predictivo
        ├── Dashboard
        └── Informe final
```

Esta organización permite separar claramente cada fase del tratamiento de los datos y facilita tanto el mantenimiento del proyecto como la incorporación de nuevas fuentes de información en el futuro, si fuese necesario.

---

## 4. Definición de la capa gold

La capa **gold** estará formada por un único conjunto de datos denominado **gold_demanda_hosteleria.parquet**.

Este dataset será el resultado final del proceso de preparación de datos y actuará como punto de partida para todas las fases posteriores del proyecto. Una vez construido, no será necesario volver a consultar directamente las fuentes originales, ya que toda la información relevante estará integrada y preparada para su utilización.

El objetivo principal de este dataset es concentrar en un único fichero la información necesaria para estudiar la relación entre las condiciones meteorológicas y la demanda en establecimientos de hostelería.

### Descripción funcional

Cada registro del dataset representará la actividad de un establecimiento durante un día concreto.

Para cada combinación de establecimiento y fecha se incluirán tanto los datos de actividad del negocio como las variables meteorológicas correspondientes a esa misma jornada.

De esta forma será posible analizar cómo determinadas condiciones climáticas pueden influir en el número de clientes o en el volumen de ventas registrado.

### Granularidad

La granularidad del dataset será:

> Una fila por establecimiento y día.

Esto significa que no se almacenará información individual de cada cliente ni de cada venta realizada.

Por ejemplo, un registro podría representar el resumen de actividad del establecimiento **A** correspondiente al día **15 de julio de 2024**.

Esta granularidad se considera suficiente para los objetivos del proyecto y simplifica considerablemente el análisis posterior.

### Volumen esperado

Tomando como referencia una cadena ficticia formada por cinco establecimientos y un histórico aproximado de tres años, el dataset contendrá alrededor de:

```text
365 días × 3 años × 5 establecimientos ≈ 5.500 registros
```

Este volumen resulta suficiente para realizar análisis exploratorios y entrenar modelos predictivos de complejidad media sin necesidad de utilizar herramientas específicas para grandes volúmenes de datos.

### Estructura del dataset

| Campo | Tipo | Descripción |
|--------|------|-------------|
| fecha | date | Fecha correspondiente al registro. |
| id_establecimiento | string | Identificador único del establecimiento. |
| municipio | string | Municipio donde se encuentra el establecimiento. |
| temperatura_maxima | float | Temperatura máxima registrada durante el día. |
| temperatura_minima | float | Temperatura mínima registrada durante el día. |
| temperatura_media | float | Temperatura media diaria. |
| precipitacion | float | Precipitación acumulada durante la jornada (mm). |
| humedad | float | Humedad relativa media (%). |
| velocidad_viento | float | Velocidad media del viento (km/h). |
| estado_cielo | string | Estado meteorológico predominante. |
| festivo | boolean | Indica si la fecha corresponde a un festivo. |
| dia_semana | string | Día de la semana. |
| numero_clientes | integer | Número total de clientes registrados. |
| numero_tickets | integer | Número de tickets emitidos. |
| importe_ventas | float | Importe total de ventas del día. |

### Clave primaria

Aunque el dataset se almacenará como un fichero y no como una base de datos relacional, cada registro podrá identificarse de forma única mediante la combinación de los campos:

```text
fecha + id_establecimiento
```

Esta combinación garantiza que únicamente exista un resumen diario para cada establecimiento.

### Variables objetivo

El proyecto contempla dos posibles variables objetivo, dependiendo del enfoque que finalmente adopte el modelo predictivo.

La primera opción consiste en predecir el **número de clientes** esperado para un día determinado. Esta variable permite estimar directamente la afluencia prevista y facilitar la planificación del personal o de la capacidad del establecimiento.

Como alternativa, también se valorará utilizar el **importe total de ventas** como variable objetivo. En este caso el modelo estaría orientado a estimar la facturación diaria prevista en función de las condiciones meteorológicas.

La decisión definitiva se tomará una vez realizado el análisis exploratorio de los datos y estudiada la calidad de ambas variables.

### Variables predictoras

Las principales variables que servirán como entrada para el modelo serán:

- Temperatura máxima.
- Temperatura mínima.
- Temperatura media.
- Precipitación.
- Humedad relativa.
- Velocidad del viento.
- Estado del cielo.
- Día de la semana.
- Indicador de festivo.

Además de estas variables, durante el desarrollo del proyecto podrán incorporarse nuevas variables derivadas si se considera que mejoran la capacidad predictiva del modelo.

Algunos ejemplos podrían ser:

- Estación del año.
- Fin de semana.
- Días consecutivos de lluvia.
- Temperatura media de los últimos días.
- Episodios de calor extremo.

### Uso del dataset

El dataset de la capa **gold** será el único conjunto de datos utilizado durante las siguientes fases del proyecto.

Concretamente, será consumido por:

| Fase | Utilización |
|------|-------------|
| Análisis exploratorio (EDA) | Estudio de relaciones entre variables y detección de patrones. |
| Ingeniería de variables | Creación de nuevas variables derivadas para mejorar el modelo. |
| Modelo predictivo | Entrenamiento y evaluación de distintos algoritmos de Machine Learning. |
| Dashboard | Visualización de la evolución histórica y de las predicciones obtenidas. |
| Informe final | Obtención de métricas, gráficos y conclusiones del proyecto. |

Definir un único dataset de referencia permitirá que todas las fases del proyecto trabajen sobre exactamente la misma información, reduciendo el riesgo de inconsistencias y facilitando la reproducibilidad del trabajo.

---

## 5. Relaciones entre los datos

El proyecto trabajará inicialmente con varios conjuntos de datos independientes, cada uno de ellos procedente de una fuente distinta. Aunque finalmente toda la información se integrará en un único dataset dentro de la capa **gold**, durante las fases de extracción y transformación será necesario mantener cada fuente por separado.

Los principales conjuntos de datos serán los siguientes:

| Dataset | Contenido |
|---------|-----------|
| meteorologia | Variables meteorológicas obtenidas de AEMET u Open-Meteo. |
| ventas | Información diaria sobre la actividad del establecimiento. |
| festivos | Calendario de festivos nacionales y autonómicos. |

### Relación entre los datasets

La relación principal del proyecto se establecerá entre los datos meteorológicos y los datos de actividad del negocio.

Para poder combinar ambas fuentes será necesario que cada registro haga referencia al mismo día y a la misma ubicación geográfica.

De forma simplificada, la relación será la siguiente:

```text
Meteorología
      │
      │ fecha + municipio
      ▼
Actividad del negocio
```

En caso de trabajar con varios establecimientos ubicados en distintos municipios, cada establecimiento estará asociado previamente a un municipio concreto. Esto permitirá asignar las condiciones meteorológicas correspondientes a cada local.

El proceso de integración seguirá, de forma general, el siguiente esquema:

```text
Establecimiento
        │
        ▼
Municipio
        │
        ▼
Datos meteorológicos
```

De esta forma, todos los registros de un establecimiento heredarán la información meteorológica correspondiente al municipio donde se encuentra ubicado.

### Tipo de relaciones

Aunque el resultado final será un único dataset, durante la fase de preparación pueden identificarse las siguientes relaciones:

| Relación | Tipo |
|----------|------|
| Establecimiento → Ventas | 1:N |
| Municipio → Datos meteorológicos | 1:N |
| Calendario → Ventas | 1:N |

Cada establecimiento podrá generar múltiples registros de ventas a lo largo del tiempo y cada municipio dispondrá de un registro meteorológico para cada fecha disponible.

### Claves de unión

Las principales claves utilizadas durante el proceso de integración serán:

- **Fecha**, para relacionar los datos correspondientes a un mismo día.
- **Municipio**, para asociar cada establecimiento con la información meteorológica adecuada.
- **Identificador del establecimiento**, para mantener la trazabilidad de los datos de actividad.

Estas claves permitirán construir el dataset final sin perder información sobre el origen de cada registro.

### Operaciones necesarias

Durante la construcción de la capa **gold** será necesario realizar diferentes operaciones de integración, entre las que destacan:

- Unión de los datos meteorológicos con los datos de actividad mediante la fecha y el municipio.
- Incorporación de la información sobre festivos a partir de la fecha.
- Creación de variables derivadas, como el día de la semana o la estación del año.
- Agregación de datos en caso de que alguna fuente presente una granularidad distinta.

Estas operaciones se realizarán durante la fase de transformación, de manera que el dataset final ya contenga toda la información necesaria para el análisis.

### Posibles dificultades

La integración de distintas fuentes puede presentar algunos problemas que deberán tenerse en cuenta durante el desarrollo del proyecto.

Uno de los principales riesgos es que existan diferencias en la cobertura temporal de las fuentes. Es posible que haya fechas para las que se disponga de datos de ventas pero no de información meteorológica, o viceversa.

También pueden aparecer diferencias en la forma de identificar las ubicaciones. Algunas fuentes pueden utilizar nombres completos de municipios, mientras que otras emplean códigos o abreviaturas. En estos casos será necesario unificar previamente la información antes de realizar la unión de los datasets.

Por último, es importante comprobar que todas las fuentes utilizan la misma granularidad temporal. Si alguna proporciona datos horarios y otra únicamente datos diarios, será necesario agregarlos antes de integrarlos en la capa **gold**.

---

## 6. Diccionario de datos inicial

A continuación se presenta un primer diccionario de datos con los campos más relevantes que formarán parte del dataset de la capa **gold**.

Este diccionario tiene como objetivo documentar el significado de cada variable, su tipo de dato esperado y la fuente de la que procede. Durante el desarrollo del proyecto podrían incorporarse nuevas variables derivadas, aunque las definidas en esta fase constituyen la base sobre la que se realizará el análisis y el entrenamiento del modelo.

| Campo | Descripción | Tipo de dato | Fuente | Obligatorio | Observaciones |
|--------|-------------|--------------|--------|-------------|---------------|
| fecha | Fecha correspondiente al registro. | date | Ventas / Meteorología | Sí | Formato ISO (YYYY-MM-DD). |
| id_establecimiento | Identificador único del establecimiento. | string | Ventas | Sí | Se utilizará junto con la fecha como identificador del registro. |
| municipio | Municipio donde se encuentra el establecimiento. | string | Ventas | Sí | Debe coincidir con la denominación utilizada en los datos meteorológicos. |
| numero_clientes | Número total de clientes registrados durante el día. | integer | Ventas | Sí | Variable candidata a ser utilizada como objetivo del modelo. |
| numero_tickets | Número de tickets emitidos durante la jornada. | integer | Ventas | No | Puede utilizarse para análisis complementarios. |
| importe_ventas | Importe total de ventas del día. | float | Ventas | Sí | Segunda posible variable objetivo del modelo. |
| temperatura_maxima | Temperatura máxima registrada. | float | AEMET | Sí | Expresada en grados Celsius. |
| temperatura_minima | Temperatura mínima registrada. | float | AEMET | Sí | Expresada en grados Celsius. |
| temperatura_media | Temperatura media diaria. | float | AEMET | Sí | Puede calcularse si la fuente no la proporciona directamente. |
| precipitacion | Precipitación acumulada durante el día. | float | AEMET | Sí | Expresada en milímetros. |
| humedad | Humedad relativa media. | float | AEMET | No | Puede no estar disponible en todas las fuentes. |
| velocidad_viento | Velocidad media del viento. | float | AEMET | No | Expresada en kilómetros por hora. |
| estado_cielo | Estado meteorológico predominante. | string | AEMET | No | Podrá requerir una normalización de categorías. |
| festivo | Indica si la fecha corresponde a un festivo. | boolean | Calendario | Sí | Variable útil para el modelo predictivo. |
| dia_semana | Día de la semana correspondiente a la fecha. | string | Derivada | Sí | Se generará automáticamente a partir de la fecha. |
| estacion | Estación del año. | string | Derivada | No | Variable derivada utilizada para capturar patrones estacionales. |
| fin_semana | Indica si el registro corresponde a sábado o domingo. | boolean | Derivada | Sí | Variable derivada para facilitar el análisis de comportamiento. |

### Variables derivadas

Además de los datos obtenidos directamente de las distintas fuentes, durante el proceso de transformación se crearán algunas variables adicionales que pueden aportar información relevante al modelo predictivo.

En esta fase se contempla, al menos, la generación de las siguientes variables:

- Día de la semana.
- Indicador de fin de semana.
- Estación del año.
- Temperatura media (si fuese necesario calcularla).
- Indicador de festivo.

La incorporación de nuevas variables dependerá de los resultados obtenidos durante el análisis exploratorio de los datos. Si se identifican patrones que puedan mejorar el rendimiento del modelo, se estudiará la posibilidad de generar nuevas características derivadas a partir de la información disponible.

### Evolución del diccionario

Este diccionario debe entenderse como una primera versión de la documentación del proyecto.

Es posible que, durante el desarrollo del análisis exploratorio o del proceso de modelado, se detecte la necesidad de incorporar nuevas variables o modificar algunas de las existentes. En caso de producirse estos cambios, el diccionario se actualizará para mantener la trazabilidad y facilitar la comprensión del conjunto de datos utilizado en cada fase del proyecto.

---

## 7. Problemas de calidad esperados

Como ocurre en la mayoría de proyectos de análisis de datos, es previsible que la información obtenida de las distintas fuentes presente problemas de calidad que deban resolverse antes de construir la capa **gold**.

Aunque los datos meteorológicos proceden de fuentes oficiales, esto no garantiza que estén completamente libres de incidencias. Además, al combinar varias fuentes diferentes pueden aparecer inconsistencias que será necesario detectar y corregir durante la fase de preparación de los datos.

A continuación se describen los principales problemas que podrían aparecer en este proyecto.

### Valores nulos

Uno de los problemas más habituales es la existencia de valores nulos en algunas variables meteorológicas.

Por ejemplo, determinadas estaciones pueden no registrar correctamente algunos datos en una fecha concreta o puede haber variables que no estén disponibles para todos los municipios.

En los datos de actividad también podrían aparecer valores faltantes si algún establecimiento no dispone de información para un determinado día.

Antes de decidir cómo tratar estos registros será necesario analizar el porcentaje de valores ausentes y estudiar si su eliminación puede afectar al resultado del análisis.

### Registros duplicados

Es posible que durante la descarga de información desde las APIs o durante el proceso de integración aparezcan registros duplicados.

Antes de construir el dataset final se comprobará que para cada combinación de fecha y establecimiento exista únicamente un registro.

En caso de detectar duplicados, se analizará su origen para decidir si deben eliminarse o consolidarse.

### Inconsistencia en nombres y categorías

Al trabajar con distintas fuentes puede ocurrir que una misma información aparezca representada de formas diferentes.

Por ejemplo, un municipio podría aparecer escrito de forma distinta según la fuente utilizada o el estado del cielo podría utilizar categorías diferentes entre AEMET y Open-Meteo.

Estas diferencias dificultan la integración de los datos y pueden provocar errores durante los cruces, por lo que será necesario normalizar previamente los valores.

### Formato de fechas

Todas las fuentes deberán utilizar un formato de fecha homogéneo.

Aunque normalmente las APIs proporcionan fechas bien estructuradas, algunos datasets abiertos pueden emplear formatos distintos o incluir información horaria cuando el análisis únicamente necesita datos diarios.

Antes de integrar la información se unificará el formato de todas las fechas.

### Diferencias en las unidades de medida

Las variables meteorológicas pueden utilizar unidades diferentes dependiendo de la fuente consultada.

Por ejemplo, la velocidad del viento podría expresarse en kilómetros por hora o en metros por segundo, mientras que la temperatura podría aparecer con distinta precisión decimal.

Antes de combinar los datos será necesario convertir todas las variables a una unidad común.

### Cobertura temporal

No todas las fuentes disponen necesariamente del mismo periodo histórico.

Es posible encontrar datos meteorológicos para fechas en las que no existan datos de actividad del negocio o, por el contrario, disponer de registros de ventas sin información meteorológica asociada.

Durante la construcción de la capa **gold** será necesario decidir cómo tratar estos casos para evitar introducir registros incompletos.

### Granularidad diferente

Otro posible problema es que las fuentes utilicen niveles de detalle distintos.

Mientras que los datos meteorológicos pueden estar disponibles por horas, los datos de ventas probablemente estarán agregados por día.

En ese caso será necesario transformar los datos horarios a una granularidad diaria antes de realizar la integración.

### Datos atípicos

También es posible encontrar valores extremos tanto en las variables meteorológicas como en los datos de actividad.

Algunos ejemplos podrían ser:

- Temperaturas excepcionalmente altas o bajas.
- Precipitaciones muy superiores a la media.
- Días con un número de clientes anormalmente elevado debido a un evento especial.
- Registros con ventas iguales a cero en días en los que el establecimiento permanecía abierto.

Estos casos no deben eliminarse automáticamente, ya que algunos pueden corresponder a situaciones reales y aportar información útil para el modelo predictivo.

### Variables no disponibles

Uno de los principales riesgos del proyecto es que algunas variables inicialmente previstas no puedan obtenerse.

Información como promociones comerciales, eventos locales o campañas especiales podría tener una influencia importante sobre la demanda, pero probablemente no estará disponible en fuentes públicas.

La ausencia de estas variables puede limitar la capacidad predictiva del modelo, aunque no impide desarrollar el proyecto.

### Integración de distintas fuentes

La construcción del dataset final dependerá de la correcta integración entre los datos meteorológicos y los datos de actividad.

Si existen diferencias en fechas, municipios o identificadores de los establecimientos, algunos registros podrían no encontrar correspondencia entre ambas fuentes.

Por este motivo será especialmente importante validar el resultado de cada proceso de unión antes de generar la capa **gold**.

### Resumen

Los problemas de calidad más relevantes que se esperan en este proyecto son:

- Valores nulos en algunas variables meteorológicas.
- Registros duplicados durante la integración.
- Diferencias en nombres de municipios y categorías meteorológicas.
- Formatos de fecha no homogéneos.
- Diferencias en las unidades de medida.
- Cobertura temporal distinta entre fuentes.
- Granularidad diferente.
- Presencia de valores atípicos.
- Variables relevantes que no estén disponibles.
- Posibles errores durante la unión de los distintos datasets.

Identificar estos problemas desde las primeras fases del proyecto permitirá definir un proceso de limpieza adecuado y construir una capa **gold** consistente y preparada para el análisis posterior.

---

## 8. Decisiones de limpieza y transformación previstas

Antes de comenzar el análisis exploratorio y el entrenamiento del modelo predictivo será necesario realizar una serie de transformaciones sobre los datos obtenidos de las distintas fuentes.

Estas decisiones se han planteado como una propuesta inicial y podrán modificarse durante el desarrollo del proyecto si el análisis de los datos muestra que existen alternativas más adecuadas.

### Tratamiento de valores nulos

En primer lugar se analizará la cantidad de valores nulos presente en cada una de las variables.

Si el porcentaje de registros afectados es muy reducido, se valorará su eliminación para evitar introducir ruido en el modelo.

En aquellos casos en los que la pérdida de información sea significativa, se estudiarán diferentes técnicas de imputación. Por ejemplo, las variables meteorológicas podrían completarse utilizando datos de estaciones cercanas o mediante estadísticas sencillas como la media o la mediana cuando resulte apropiado.

La decisión dependerá del tipo de variable y de la importancia que tenga para el análisis posterior.

### Eliminación de registros duplicados

Antes de construir la capa **gold** se comprobará que no existan registros duplicados.

La combinación formada por la fecha y el identificador del establecimiento deberá ser única para cada fila del dataset.

Si se detectan duplicados, primero se investigará su origen para determinar si se trata de un error en la extracción de datos o de registros válidos que deban agregarse antes de eliminarse.

### Normalización de formatos

Uno de los primeros pasos del proceso de limpieza será unificar el formato de todas las variables.

En particular, se normalizarán:

- Las fechas, utilizando el formato ISO (`YYYY-MM-DD`).
- Los nombres de las columnas para mantener una nomenclatura homogénea.
- Los nombres de municipios y otras categorías de texto.
- Las unidades de medida de las variables meteorológicas cuando sea necesario.

Este proceso facilitará la integración entre las distintas fuentes de información.

### Conversión de tipos de datos

También se revisará que cada columna utilice el tipo de dato más adecuado.

Por ejemplo:

- Las fechas se convertirán al tipo `date` o `datetime`.
- Las variables numéricas utilizarán tipos enteros o decimales según corresponda.
- Las variables categóricas se almacenarán como texto o categorías.
- Los indicadores lógicos, como el campo `festivo`, se convertirán a tipo booleano.

Realizar estas conversiones desde las primeras fases del proyecto ayudará a evitar errores durante el análisis y mejorará la eficiencia del procesamiento.

### Creación de variables derivadas

Una vez limpios los datos se crearán nuevas variables que puedan aportar información adicional al modelo predictivo.

Inicialmente se plantea generar las siguientes:

- Día de la semana.
- Indicador de fin de semana.
- Estación del año.
- Mes del año.
- Temperatura media, si no está disponible directamente.
- Indicador de festivo.

Durante el análisis exploratorio también se evaluará la utilidad de crear nuevas variables derivadas, como medias móviles de temperatura o indicadores de episodios meteorológicos extremos.

### Integración de las fuentes

Tras la limpieza individual de cada dataset se procederá a su integración.

La unión entre los datos meteorológicos y los datos de actividad se realizará utilizando principalmente la fecha y el municipio correspondiente a cada establecimiento.

Posteriormente se añadirá la información procedente del calendario de festivos para completar el conjunto de datos definitivo.

Antes de generar la capa **gold** se comprobará que todos los registros hayan sido correctamente asociados y que no existan pérdidas significativas de información durante el proceso de unión.

### Agregaciones

En caso de que alguna de las fuentes proporcione información con una granularidad distinta a la utilizada en el proyecto, será necesario realizar procesos de agregación.

Por ejemplo, si los datos meteorológicos estuvieran disponibles a nivel horario, se calcularán valores diarios como:

- Temperatura máxima.
- Temperatura mínima.
- Temperatura media.
- Precipitación acumulada.
- Humedad media.

De esta forma todas las fuentes compartirán la misma granularidad antes de ser integradas.

### Criterios para considerar un registro válido

No todos los registros disponibles se incorporarán automáticamente a la capa **gold**.

Como criterio general, un registro será considerado válido cuando:

- Disponga de una fecha correctamente identificada.
- Pueda asociarse a un establecimiento concreto.
- Tenga información meteorológica suficiente para el análisis.
- No presente errores evidentes en las variables principales.
- No corresponda a un registro duplicado.

En caso contrario, el registro será revisado individualmente y se decidirá si puede recuperarse mediante algún proceso de transformación o si debe descartarse.

### Datos que podrán descartarse

Durante el proceso de preparación también podrán eliminarse aquellos datos que no aporten valor al objetivo del proyecto.

Entre ellos se incluyen:

- Registros completamente duplicados.
- Variables que no tengan utilidad para el análisis o el modelo.
- Columnas técnicas generadas por las APIs que no contengan información relevante.
- Registros claramente erróneos que no puedan corregirse de forma fiable.

El objetivo de estas decisiones es construir un dataset final limpio, consistente y fácil de interpretar, evitando incorporar información que pueda afectar negativamente al rendimiento del modelo predictivo.

---

## 9. Riesgos del modelo de datos

En términos generales, considero que el modelo de datos planteado es adecuado para el alcance del proyecto y permite desarrollar todas las fases previstas, desde el análisis exploratorio hasta el entrenamiento del modelo predictivo y la creación del dashboard final.

Uno de los aspectos que tengo más claros es la estructura de las distintas capas de datos. La separación entre las capas **raw**, **processed** y **gold** permitirá mantener un flujo de trabajo ordenado y facilitará tanto la reproducibilidad como el mantenimiento del proyecto. Además, la definición de un único dataset final simplifica el desarrollo de las fases posteriores y evita trabajar con múltiples versiones de la información.

La mayor incertidumbre del proyecto se encuentra en la obtención de los datos de actividad del negocio. Mientras que los datos meteorológicos pueden obtenerse fácilmente a través de fuentes oficiales como AEMET, la información sobre ventas o número de clientes suele pertenecer a empresas privadas y no está disponible públicamente con el nivel de detalle necesario.

En caso de no encontrar un conjunto de datos adecuado, la alternativa será generar un dataset sintético que reproduzca un comportamiento realista a partir de patrones observados en estudios, datos abiertos y la información meteorológica disponible. Aunque esta solución no sustituye completamente a los datos reales, permitirá desarrollar todas las fases del proyecto y evaluar la viabilidad técnica del modelo propuesto.

Otro riesgo importante está relacionado con la integración de las distintas fuentes. Será necesario comprobar que las fechas, municipios y demás variables utilizadas como claves de unión sean consistentes entre los diferentes datasets. Cualquier discrepancia podría provocar pérdidas de información o asociaciones incorrectas durante la construcción de la capa **gold**.

También existe la posibilidad de que algunas variables inicialmente previstas, como información sobre promociones, eventos locales o campañas comerciales, no puedan incorporarse al modelo por falta de disponibilidad. Estas variables podrían influir en la demanda de los establecimientos y su ausencia puede limitar parcialmente la capacidad predictiva del modelo.

Si finalmente no fuese posible construir la capa **gold** exactamente como se ha definido en esta entrega, el proyecto podría simplificarse reduciendo el número de variables utilizadas o limitando el análisis a aquellas fuentes cuya calidad y disponibilidad estén garantizadas. Por ejemplo, podría desarrollarse un modelo basado únicamente en la relación entre temperatura, precipitación y demanda diaria, incorporando el resto de variables en futuras ampliaciones del proyecto.

En conjunto, considero que los riesgos identificados son asumibles dentro del contexto de un proyecto académico. La mayor parte de ellos están relacionados con la disponibilidad y calidad de los datos, más que con limitaciones técnicas. Definir estos posibles escenarios desde esta fase permitirá adaptar el desarrollo del proyecto si aparecen dificultades, manteniendo siempre el objetivo principal de analizar la influencia de la meteorología sobre la demanda en establecimientos de hostelería.

---

## Valoración final

Con esta entrega queda definida una primera versión del modelo de datos que servirá como base para el desarrollo del proyecto durante las siguientes fases del curso. Aunque algunos aspectos podrán ajustarse a medida que avance el trabajo y se conozcan mejor los datos disponibles, considero que la estructura planteada proporciona una base sólida sobre la que construir el análisis, el modelo predictivo y las visualizaciones finales.

Mis expectativas para las próximas entregas son poder comenzar a trabajar con datos reales lo antes posible, validar las hipótesis planteadas y comprobar hasta qué punto las variables meteorológicas influyen realmente en la demanda de los establecimientos. También espero que el análisis exploratorio permita descubrir relaciones que inicialmente no había considerado y que puedan mejorar el rendimiento del modelo.

Soy consciente de que pueden surgir dificultades, especialmente en la obtención de datos de actividad del negocio o en la integración de las distintas fuentes. Aun así, creo que existen alternativas viables, como el uso de datos sintéticos o datasets públicos, que permitirán continuar con el desarrollo del proyecto sin perder de vista los objetivos principales.

En conjunto, esta fase me ha servido para planificar con mayor detalle cómo se organizarán los datos y cómo será el flujo de trabajo durante el resto del proyecto. Espero que esta planificación facilite las siguientes etapas y permita centrar el esfuerzo en el análisis y en la construcción de un modelo que aporte resultados útiles e interesantes.