# Entrega 4 · Análisis y modelado

**Por Juan Francisco Morales Olivo**

---

## 1. Qué se quiere predecir y para qué

El objetivo es estimar el **número de clientes que tendrá el restaurante en un día concreto**, con la información disponible antes del servicio.

Es un problema de **regresión**: la variable objetivo es numérica y continua.

Pero conviene no perder de vista para qué sirve la cifra. No se predice por predecir: se predice para decidir **cuánta gente hay que llamar ese día**. Esa finalidad condiciona tres cosas del modelado:

1. **Qué métrica importa.** El error se mide en clientes, no en unidades abstractas, porque es lo que el encargado tiene que traducir en personas.

2. **Qué variables se pueden usar.** Solo las que se conocen antes del servicio. Si el modelo necesita saber cómo fue el día para predecir cómo va a ir, no sirve.

3. **Qué se entrega.** Una cifra sola no vale. Hace falta un rango, porque planificar con la predicción puntual deja el negocio expuesto la mitad de los días.

---

## 2. Datos de entrada

El dataset es la capa **gold** del proyecto (`data/gold/demanda_restaurante.csv`), construida en la Entrega 3.

**1.129 días de actividad**, del 1 de enero de 2022 al 31 de julio de 2026. Cada fila es un día en el que el restaurante abrió.

Los días de cierre —lunes y martes de descanso, todo agosto, Navidad— **no están en el dataset**. Un día cerrado no es un día de cero clientes: es un día sin observación. Incluirlos hundiría la media y enseñaría al modelo un patrón que no existe.

### 2.1. Variable objetivo

`n_clientes`: número de clientes atendidos ese día.

| Estadístico | Valor |
|---|---|
| Media | 100,3 clientes |
| Mínimo | 28 |
| Máximo | 273 |

La distribución no es una campana. Son dos poblaciones mezcladas: los días entre semana en torno a 50-70 clientes, y los fines de semana muy por encima. Separar esas dos poblaciones es la mayor parte del trabajo del modelo.

### 2.2. Variables de entrada

Nueve variables, definidas en un único sitio: `src/models/features.py`.

**Categóricas** (codificadas con One-Hot):

| Variable | Descripción |
|---|---|
| `dia_semana` | Día de la semana (0 = lunes … 6 = domingo) |
| `mes` | Mes del año |
| `tipo_vacaciones` | ninguna / navidad / semana_santa / verano / otros |

`dia_semana` y `mes` son números, pero se tratan como categorías. El efecto del sábado no es "seis veces" el del lunes, ni el de diciembre "doce veces" el de enero.

**Numéricas y binarias:**

| Variable | Descripción |
|---|---|
| `es_festivo` | Festivo nacional, autonómico o local |
| `tmed` | Temperatura media (°C) |
| `amplitud_termica` | `tmax − tmin` (°C) |
| `prec_log` | `log(1 + precipitación)` |
| `lluvia_fin_semana` | Interacción: `prec_log × fin_de_semana` |

**Disponibles pero apagadas**, con un interruptor en `features.py`:

| Variable | Descripción | Por qué está apagada |
|---|---|---|
| `mes_finde` | Interacción mes × fin de semana (24 niveles) | Medida: no mejora (§3.3) |
| `es_vispera_festivo` | El día siguiente es festivo | Inerte en datos sintéticos |
| `es_puente` | Laborable encajonado entre festivo y fin de semana | Inerte en datos sintéticos |

Están construidas y probadas. Con datos reales de un local son lo primero que habría que volver a medir.

---

## 3. Las cinco decisiones que definieron el conjunto de variables

Esta parte es la que más tiempo llevó y la que más condiciona el resultado. Cada exclusión tiene un motivo medible.

### 3.1. Nada que solo se sepa después del servicio

`nota_faena` describe cómo fue la jornada: si faltó personal, si sobró. Es un **resultado**, no una entrada. Usarla sería fuga de información: el modelo acertaría en la evaluación y sería inútil en producción, porque el jueves por la tarde nadie sabe todavía cómo va a ir el sábado.

### 3.2. Fuera las variables colineales

La capa gold contiene varias variables que son función exacta de otras que ya están:

| Variable excluida | Es exactamente |
|---|---|
| `fin_de_semana` | `dia_semana ∈ {5, 6}` |
| `es_vacaciones` | `tipo_vacaciones ≠ "ninguna"` |

Mantenerlas no añade información y desestabiliza los coeficientes de la regresión lineal, que es justamente el modelo que queremos poder interpretar.

El síntoma era visible en una versión anterior del proyecto: `es_semana_santa` y `tipo_vacaciones_semana_santa` salían con coeficientes idénticos hasta el décimo decimal (16,375347204145836 y 16,37534720414579). Esa coincidencia es la firma de la colinealidad perfecta: el modelo no puede repartir el efecto entre dos columnas que son la misma.

### 3.3. Las interacciones: planteadas, medidas y descartadas

Esta es la decisión que más cambió respecto a la primera versión del proyecto, y la que mejor ilustra la diferencia entre un argumento y una medición.

**El argumento.** Un sábado de junio no se comporta como la suma de "sábado" y "junio": hay terraza, comuniones y celebraciones de fin de curso. Un modelo lineal no puede representar eso solo con efectos principales, así que hace falta una interacción. El argumento es correcto.

**Lo que estaba mal.** La primera versión lo resolvió haciendo que el calendario publicase `es_fin_semana_mayo`, `es_fin_semana_junio` y `es_fin_semana_julio`. Dos problemas, y el segundo es grave:

1. *Tres meses elegidos a mano.* Si la hipótesis es que la interacción existe, hay que plantearla para los doce meses y dejar que los datos digan cuáles importan. Con tres, lo que se mide es la corazonada, no el efecto.

2. *Elegidos por el motivo equivocado.* Mayo, junio y julio son exactamente los tres meses que el generador sintético amplifica. El calendario estaba copiando el proceso generador en lugar de describir el almanaque. Eso es fuga conceptual: información sobre cómo se fabricaron los datos, colada en la capa de datos.

   Y ni siquiera eran los mejores candidatos. La ratio findes/laborables más alta de la serie está en **noviembre (1,955)** y **abril (1,913)**, por encima de mayo (1,839).

**Cómo se resolvió.** La interacción se construye ahora en `features.py`, para los doce meses (`mes_finde`, 24 niveles), y se compara contra el conjunto de efectos principales en `src/analysis/comparar_variables.py`. Con tres medidas, porque una sola engaña:

| Variante | Columnas | MAE validación | MAE test | MAE val. cruzada |
|---|---|---|---|---|
| **A · Solo efectos principales** | **23** | **16,19** | **17,01** | **16,48 ± 1,77** |
| B · + interacción mes × finde | 44 | 15,36 | 16,59 | 16,04 ± 1,64 |
| C · + vísperas y puentes | 25 | 16,18 | 17,02 | 16,51 ± 1,76 |
| D · Todo encendido | 46 | 15,36 | 16,59 | 16,08 ± 1,65 |

El criterio fijado de antemano: una variante gana si mejora el MAE de validación cruzada en **más de una desviación típica**. El umbral es 16,48 − 1,77 = **14,71**. Ninguna baja de ahí.

**Conclusión: se despliega A.** Veintiuna columnas más para una diferencia que cabe dentro del ruido no es un intercambio razonable.

Y conviene ser preciso sobre lo que ese resultado dice. **No** dice que la interacción no exista: sabemos que existe, porque el generador la produce (entre un 5 % y un 8 % en los fines de semana de mayo, junio y julio). Dice que un efecto de ese tamaño, con unas 1.100 observaciones, es más pequeño que el error de estimar las columnas que hacen falta para capturarlo. Con datos reales de un local, donde el efecto terraza puede ser mucho mayor, la medición hay que repetirla — y el interruptor está puesto para eso.

### 3.4. La temperatura, como nivel y amplitud

Este fue el hallazgo más importante del análisis.

Al calcular el factor de inflación de la varianza (VIF) de las tres temperaturas, el resultado fue absurdo:

| Variable | VIF |
|---|---|
| `tmed` | 45.791 |
| `tmax` | 16.169 |
| `tmin` | 9.198 |

Por encima de 10 ya se considera un problema serio.

La causa se comprobó directamente sobre los datos: **AEMET no mide la temperatura media, la calcula** como `(tmax + tmin) / 2`. La identidad se cumple con una desviación máxima de 0,05 °C, que es puro redondeo.

Es decir: las tres columnas solo tienen **dos grados de libertad**. La tercera es combinación lineal exacta de las otras dos.

El efecto sobre la regresión lineal era demoledor:

| Variable | Coeficiente |
|---|---|
| temperatura media | −104,2 |
| temperatura máxima | +58,1 |
| temperatura mínima | +56,8 |

Ninguno de esos números significa nada por separado: se anulan entre sí. Un modelo así no se puede explicar a nadie.

**La solución fue cambiar de parametrización sin perder información:**

```
tmed              →  nivel térmico del día
amplitud_termica  →  tmax − tmin
```

La transformación es invertible (`tmax = tmed + amplitud/2`, `tmin = tmed − amplitud/2`), así que los modelos de árboles conservan toda la información y pueden seguir encontrando los umbrales de calor y frío extremo. Pero la correlación entre ambas baja a 0,47 y **el VIF a 1,28**.

Con la nueva parametrización los coeficientes vuelven a ser legibles, y además la importancia por permutación de la temperatura pasó de un valor artificialmente inflado (105,9) a su valor real (2,05).

### 3.5. `n_empleados` fuera del modelo

Es la decisión menos obvia y la más importante para el producto.

`n_empleados` correlaciona bien con `n_clientes`, y la tentación de usarla es grande. Pero:

**No aporta información nueva.** La plantilla se decide a partir del día de la semana: los sábados se ponen más camareros. El modelo ya tiene `dia_semana`, así que `n_empleados` solo se lo repite con ruido añadido.

**Y haría el producto circular.** La dirección útil para el negocio es la contraria:

> No quiero predecir clientes a partir de la plantilla. Quiero predecir clientes **para decidir** la plantilla.

Si el modelo necesitara saber cuántos empleados hay para estimar cuántos clientes vienen, el encargado tendría que haber tomado ya la decisión que el sistema debería ayudarle a tomar.

Por eso `n_empleados` se reserva para la **salida**: la aplicación estima clientes y a partir de ahí recomienda plantilla, usando los mismos umbrales de clientes por empleado con los que se etiqueta la jornada en los datos de actividad.

---

## 4. Separación de los datos

La separación es **temporal**, nunca aleatoria:

| Conjunto | Periodo | Registros | Para qué |
|---|---|---|---|
| Entrenamiento | 2022, 2023, 2024 | 725 | El modelo aprende |
| Validación | 2025 | 246 | Se compara y se elige el modelo |
| Test | 2026 | 158 | Estimación final; se toca una vez |

Una separación aleatoria mezclaría días de 2026 en el entrenamiento y días de 2022 en el test. El modelo estaría aprendiendo del futuro y el error resultante sería mucho mejor que el que tendría funcionando de verdad, donde solo se conoce el pasado.

En un problema temporal la separación aleatoria no es una simplificación aceptable: es un error metodológico que invalida la evaluación.

**El conjunto de test se usó una sola vez, al final.** Toda la comparación y toda la elección de modelo se hicieron sobre validación.

---

## 5. Modelos planteados

### 5.1. Baseline

Antes de cualquier modelo hace falta una referencia. El baseline predice, para cada día, **la media histórica de clientes de ese día de la semana**.

No es machine learning: es lo que haría un encargado con experiencia y sin ninguna herramienta. *"Los sábados suelen venir unos 150, pues pongo para 150."*

Si un modelo no supera claramente esto, no aporta valor y no merece la pena mantenerlo por muy sofisticado que sea.

Medias aprendidas del periodo 2022-2024:

| Día | Clientes |
|---|---|
| Miércoles | 51 |
| Jueves | 71 |
| Viernes | 103 |
| Sábado | 151 |
| Domingo | 120 |

### 5.2. Regresión lineal

El primer modelo de machine learning. Se elige por interpretabilidad: cada coeficiente dice cuántos clientes suma o resta una variable, y eso se puede explicar al responsable del negocio.

Las variables numéricas se estandarizan para que los coeficientes sean comparables entre sí.

### 5.3. Random Forest

Primer modelo no lineal. Aporta dos cosas que la regresión no puede dar: relaciones no lineales (la demanda sube con la temperatura hasta un punto y vuelve a bajar con el calor extremo) e interacciones automáticas.

A cambio pierde interpretabilidad directa y no puede extrapolar fuera del rango que ha visto.

Hiperparámetros conservadores (500 árboles, profundidad máxima 12, mínimo 3 muestras por hoja) porque el conjunto de entrenamiento es pequeño.

### 5.4. Hist Gradient Boosting

Boosting por histogramas. A diferencia del Random Forest, entrena los árboles en cadena: cada uno corrige el error del anterior. Suele ser el modelo más preciso en problemas tabulares.

Se usa `early_stopping` con validación interna para que deje de entrenar cuando ya no mejora.

### 5.5. Protocolo común

Los cuatro modelos pasan **exactamente por el mismo proceso**: los mismos conjuntos, las mismas variables, las mismas métricas y el mismo formato de resultados. Está implementado en `src/models/entrenamiento.py`.

Esto importa más de lo que parece. En una versión anterior del proyecto, cuatro ficheros distintos entrenaban cada uno su propio modelo con variables y particiones diferentes, y las cifras que se publicaban no correspondían al modelo que se evaluaba. Con el protocolo común, la única diferencia entre los modelos es el algoritmo.

---

## 6. Métricas

| Métrica | Qué mide | Por qué está |
|---|---|---|
| **MAE** | Error medio absoluto, en clientes | **Métrica principal.** Se lee directamente en las unidades del negocio |
| **RMSE** | Penaliza más los errores grandes | Detecta días de fallo grave, que son los que dejan al local sin personal |
| **MAPE** | Error relativo | Permite comparar días de mucha y poca afluencia |
| **R²** | Variabilidad explicada | Contexto general del ajuste |

El MAE es la métrica principal porque es la que se traduce en una decisión: "el modelo se equivoca de media en 15 clientes" es una frase que el encargado puede usar. "El R² es 0,75" no lo es.

---

## 7. Resultados

### 7.1. Comparación

| Modelo | MAE validación | RMSE validación | MAE test | RMSE test | MAPE test | R² test |
|---|---|---|---|---|---|---|
| **Regresión lineal** | 16,19 | 22,19 | **17,01** | **23,81** | 17,1 % | **0,736** |
| Hist Gradient Boosting | **16,02** | 22,56 | 17,67 | 24,42 | 16,2 % | 0,723 |
| Random Forest | 16,73 | 24,05 | 17,31 | 25,71 | 16,1 % | 0,692 |
| Baseline | 20,58 | 27,71 | 20,61 | 28,90 | 19,2 % | 0,611 |

**La mejora sobre el baseline es del 21 % en validación y del 17 % en test.** Es real pero moderada, y tiene una explicación clara: el día de la semana explica la mayor parte de la demanda y el baseline ya lo usa. Lo que aportan los modelos es lo que queda —estacionalidad, festivos, meteorología—, que es menos de lo que la intuición sugiere.

### 7.2. ¿Es real la diferencia entre modelos?

Los tres modelos de machine learning quedan muy cerca. Elegir el mejor por una sola partición sería arriesgado: la diferencia puede ser menor que el ruido.

Para comprobarlo se repitió la comparación cinco veces sobre periodos distintos, siempre entrenando con el pasado y evaluando con el futuro inmediato (validación cruzada temporal con origen deslizante, bloques de 120 días):

| Modelo | MAE medio | Desviación | Bloques |
|---|---|---|---|
| **Regresión lineal** | **16,48** | 1,77 | 13,8 · 19,1 · 17,3 · 15,5 · 16,7 |
| Hist Gradient Boosting | 16,52 | **0,92** | 15,7 · 16,5 · 18,0 · 15,4 · 16,9 |
| Random Forest | 17,35 | 1,10 | 16,5 · 17,9 · 18,9 · 15,8 · 17,7 |
| Baseline | 19,19 | 1,76 | 16,0 · 19,3 · 19,8 · 21,5 · 19,4 |

**Empatan, y no por poco: empatan del todo.** La regresión lineal tiene el menor error medio por 0,03 clientes (16,48 frente a 16,52) y gana en tres de los cinco bloques; el boosting es bastante más estable (0,92 de desviación frente a 1,77). Cada una gana en una cosa distinta y ninguna de las dos diferencias es interpretable.

Para no quedarse en la impresión, se hizo la comparación pareada de errores absolutos día a día. En validación, la diferencia entre ambos modelos es de **+0,17 clientes con un intervalo de confianza del 95 % de [−1,26, +1,60]**; en test, de −0,66 con [−2,57, +1,25]. Los dos intervalos contienen el cero holgadamente.

Esto cambia la naturaleza de la decisión, y conviene decirlo sin adornos: **el criterio declarado de antemano no desempata**. Por MAE de validación gana el boosting por 0,17 clientes, que es ruido. Quien elija por esa cifra está eligiendo al azar. Lo que desempata son los motivos secundarios, que también estaban declarados antes de ver los resultados (§7.3).

También merece la pena mirar la columna de la desviación. La regresión es la más **estable** de las cuatro (0,86 frente a 1,15 del boosting y 1,30 del bosque): rinde de forma parecida en periodos distintos, que es justo lo que hace falta en un sistema que va a funcionar mes tras mes sin supervisión.

### 7.3. Modelo seleccionado

**Regresión lineal**, y la justificación ha cambiado respecto a la primera versión de este documento, así que conviene contarla entera.

El criterio declarado antes de ver los resultados era el MAE en validación. Por esa cifra gana el **boosting**, por 0,17 clientes. Pero la comparación pareada de errores día a día da un intervalo de confianza del 95 % de [−1,26, +1,60]: la diferencia es indistinguible de cero. **El criterio, aplicado con rigor, no desempata.** Elegir por esa cifra sería elegir al azar y llamarlo método.

Así que desempatan los motivos secundarios, que también estaban declarados de antemano:

- **Es interpretable.** Sus coeficientes se leen directamente y se pueden contar en una reunión. La aplicación explica *por qué* sale ese número, y eso no es un adorno: un encargado que no entiende la cifra no la usa.
- **Generaliza mejor.** En test, que es el conjunto que nadie ha tocado, gana la regresión: 17,01 frente a 17,67, y R² de 0,736 frente a 0,723.
- **Se reentrena en menos de un segundo**, lo que hace viable el reentrenamiento periódico que —como muestra el apartado 7.6— es la única corrección del sesgo que funciona de verdad.

Y hay que decir lo que *no* la favorece: **la estabilidad entre periodos, que antes era su mejor argumento, ahora es del boosting** (0,92 de desviación frente a 1,77). Es el motivo secundario que cambia de bando, y el más honesto de reconocer.

Merece subrayarse el resultado, que no es el que se esperaba al empezar: **los tres modelos de aprendizaje automático son equivalentes**. La diferencia entre un modelo de 1805 y uno de 2017 es, en este problema, ruido. Lo que separa a la referencia sin modelo del resto sí es real —cuatro clientes y medio de error— pero entre ellos no hay nada que elegir por precisión.

### 7.4. Modelo desplegado

El modelo que usa la aplicación **no es el mismo objeto** que el de la tabla comparativa.

El protocolo de comparación entrena con 2022-2024 para poder evaluar con 2025 de forma limpia. Pero antes de desplegar se vuelve a entrenar con **todo el histórico disponible (2022-2025)**: los hiperparámetros y el algoritmo ya están decididos, así que no hay riesgo de elegir mirando el test, y cuanto más reciente sea el último dato que ha visto, mejor predice.

2026 sigue sin usarse para entrenar, de modo que sus métricas siguen siendo una estimación limpia:

| Modelo | Entrenado con | MAE en 2026 |
|---|---|---|
| De selección | 2022-2024 | 17,01 |
| **Desplegado** | 2022-2025 | **16,07** |

---

## 8. Qué variables usa realmente el modelo

Se mide con **importancia por permutación**: se desordena al azar una variable y se mide cuánto empeora el error. Si al romperla el modelo apenas se resiente, esa variable no le estaba aportando nada.

Es la única medida comparable entre algoritmos distintos, porque no depende del funcionamiento interno de cada uno. Se calcula sobre validación, no sobre entrenamiento.

| Variable | Regresión lineal | Random Forest | Hist Gradient Boosting |
|---|---|---|---|
| **día de la semana** | **27,02** | **22,76** | **26,69** |
| mes | 3,60 | 1,35 | 1,64 |
| temperatura media | 2,05 | 3,26 | 5,32 |
| precipitación (log) | 1,44 | 0,69 | 0,98 |
| tipo de vacaciones | 0,99 | 0,21 | 0,32 |
| fin de semana en mayo | 0,74 | 0,65 | 0,13 |
| lluvia en fin de semana | 0,71 | 0,15 | 0,16 |
| fin de semana en Semana Santa | 0,47 | 0,29 | 0,00 |

*Aumento del error medio, en clientes, al desordenar cada variable.*

**Los tres modelos coinciden**, y eso hace la conclusión mucho más sólida que si dependiera de un solo algoritmo:

**El día de la semana lo domina todo.** Su importancia es un orden de magnitud mayor que la de cualquier otra variable.

La meteorología aparece, pero muy por detrás. Esto obliga a **redimensionar la hipótesis de partida del proyecto**: la Entrega 1 planteaba que el clima condiciona la afluencia en hostelería. Los datos lo confirman —la lluvia reduce la demanda y el calor extremo también—, pero su peso es de segundo orden frente al calendario. Un sistema honesto tiene que reflejarlo en lugar de vender el efecto meteorológico como mayor de lo que es.

### 8.1. Coeficientes de la regresión lineal

Ahora que la colinealidad está resuelta, se leen directamente. Cada valor es el número de clientes que suma o resta esa variable respecto a la categoría de referencia (lunes, enero):

| Variable | Coeficiente |
|---|---|
| día de la semana: sábado | +105,7 |
| día de la semana: domingo | +74,5 |
| día de la semana: viernes | +59,2 |
| mes: diciembre | +31,9 |
| día de la semana: jueves | +30,9 |
| mes: octubre | +27,7 |
| mes: mayo | +23,1 |
| mes: abril | +20,7 |
| mes: junio | +18,2 |

**Un sábado trae unos 105 clientes más que un lunes**, manteniendo todo lo demás constante. Diciembre suma 32 sobre enero. Estas son frases que se pueden decir en voz alta delante del responsable del restaurante, y esa es exactamente la razón de haber peleado la colinealidad.

---

## 9. Análisis de errores

Una métrica global esconde tanto como enseña. Un error medio de 17 clientes puede ser 17 todos los días o 80 en cuatro sábados. Para un restaurante son situaciones muy distintas.

### 9.1. El sesgo sistemático

Este es el hallazgo más importante de la evaluación, y también el más incómodo.

| Medida | Valor |
|---|---|
| Error medio (real − predicho) | **+11,3 clientes** |
| Error absoluto medio | 17,0 clientes |
| Días subestimados | 69 % |
| Proporción del error que es sesgo | ~66 % |

**El error no es simétrico.** Dos tercios de él no es ruido: es sesgo. El modelo se queda corto de forma sistemática.

**La causa:** el negocio crece un 4 % anual. El modelo aprende del pasado y tiene que predecir el futuro, y ninguna de sus variables le dice que ha pasado el tiempo. Predice el nivel de demanda que aprendió, no el actual.

#### Corregirlo: cuatro opciones, dos horizontes y un resultado incómodo

La primera versión de este documento decía que era "la limitación más relevante y la que primero habría que atacar". Se atacó, con el método de siempre: se plantearon cuatro opciones y se midieron (`src/analysis/correccion_tendencia.py`). Con **dos horizontes**, porque extrapolar no cuesta lo mismo a unos meses que a dos años.

**Horizonte corto** — entrenando hasta 2025, que es lo que hace el modelo desplegado:

| Opción | MAE | Sesgo | Plantilla OK | Falta | Exceso | Coste |
|---|---|---|---|---|---|---|
| **A · Sin corrección** | 16,07 | +8,83 | **94,0 %** | **4** | 6 | **1.196 €** |
| B · Índice temporal | 16,01 | −3,40 | 86,2 % | 1 | 22 | 2.863 € |
| C · Ventana móvil (730 días) | 16,31 | **+2,88** | 92,8 % | 2 | 10 | 1.396 € |
| D · Corrección de nivel | **15,31** | +5,30 | 93,4 % | 2 | 9 | 1.284 € |

**Horizonte largo** — entrenando hasta 2024, o sea año y medio sin reentrenar:

| Opción | MAE | Sesgo | Plantilla OK | Falta | Exceso | Coste |
|---|---|---|---|---|---|---|
| A · Sin corrección | 17,01 | +11,25 | 93,4 % | 6 | 5 | 1.410 € |
| B · Índice temporal | 16,85 | −6,42 | 83,8 % | 1 | 26 | 3.934 € |
| **C · Ventana móvil (730 días)** | 16,16 | +7,98 | **94,6 %** | 4 | 5 | **1.076 €** |
| D · Corrección de nivel | **15,56** | +6,34 | 94,0 % | 2 | 8 | 1.458 € |

**Lo primero que salta: el MAE y la decisión no coinciden.** La opción D gana en error medio en los dos horizontes, 15,31 y 15,56, casi un punto por debajo de todo lo demás. Y no es la que mejor decide en ninguno de los dos. Si este proyecto se hubiera evaluado solo por MAE —como se evalúa la mayoría— habría desplegado D y habría tomado peores decisiones creyendo que mejoraba. Es la demostración más limpia de la tesis del trabajo.

**Lo segundo: la tendencia extrapolable es mala idea.** La opción B hace lo que promete sobre el papel —elimina el sesgo— pero lo hace pasándose de largo. El sesgo se vuelve negativo, los días de exceso saltan de 6 a 22 y de 5 a 26, y la decisión cae al 86,2 % y al 83,8 %. El coste casi triplica.

La razón es que una recta extrapolada supone que el crecimiento continúa igual, y no continúa:

| | sesgo sin tendencia | sesgo con tendencia |
|---|---|---|
| entrena 2024, evalúa 2025 | +8,53 | −3,14 |
| entrena 2025, evalúa 2026 | +8,83 | −3,40 |
| entrena 2024, evalúa 2026 | +11,25 | −6,42 |

En ninguno de los tres casos acierta: cambia un error sistemático por otro de signo contrario. Y hay un motivo adicional para desconfiar de ella, que la teoría ya anticipaba: **solo funciona en el modelo que puede extrapolar**. Un árbol no puede predecir fuera del rango que ha visto, así que con Random Forest o boosting la variable reduce el sesgo a medias y no mejora el error.

**Lo tercero, y es la respuesta: reentrenar.** La ventana móvil no aporta nada con horizonte corto —el modelo ya está al día— pero con horizonte largo es la mejor opción de la tabla: 94,6 % de plantilla correcta frente al 93,4 % de no hacer nada, y el menor coste de las ocho filas.

Dicho de otro modo: **el sesgo no se arregla modelándolo, se arregla no dejando envejecer el modelo.** Por eso el interruptor `USAR_TENDENCIA_TEMPORAL` queda apagado, y por eso existe `src/analysis/monitorizacion.py`, que vigila el sesgo en ventanas de cuatro semanas y avisa cuando toca reentrenar.

Mientras tanto, la limitación **se avisa en la propia aplicación**, con la recomendación práctica de tirar hacia el extremo alto del rango cuando la decisión sea ajustada.

### 9.2. Error por día de la semana

En términos absolutos el modelo falla más los días de mucho volumen. Pero esa comparación no es justa: es más fácil fallar 25 clientes sobre 200 que sobre 50.

En términos **relativos** la imagen se invierte: el modelo es proporcionalmente **más fiable en los días grandes**, que son precisamente los que más se juega el restaurante en la decisión de plantilla.

### 9.3. Intervalo de predicción

La aplicación no muestra una cifra sola. Muestra un **intervalo del 80 %**, calculado con los residuos empíricos del último año completo. No se asume que el error siga una distribución normal: se toman directamente los percentiles observados.

**Pero un intervalo de anchura fija no sirve aquí.** La incertidumbre de este problema no es constante ni de lejos: la desviación de los residuos pasa de 11,0 clientes en los días flojos a 30,1 en los días fuertes. Un intervalo único reparte la misma anchura a todos, y el resultado es que miente en las dos direcciones a la vez:

| Predicción | Cobertura con intervalo fijo | Con intervalo escalado |
|---|---|---|
| menos de 60 | 100,0 % | 72,7 % |
| 60-90 | 85,7 % | 85,7 % |
| 90-120 | 86,1 % | 88,9 % |
| 120-160 | 73,8 % | 76,2 % |
| más de 160 | 57,1 % | 71,4 % |

El intervalo fijo **sobra** en los días tranquilos, donde cubre el 100 % y no informa de nada, y **falta** justo en los días llenos, donde cubre el 57 % y es cuando de verdad importa saber hasta dónde puede llegar la cosa. Un encargado que se fía de él se lleva el susto precisamente los sábados.

La solución es una **regresión cuantílica** de los residuos sobre la predicción: en lugar de un número por cada extremo, una recta. La anchura pasa así de unos 38 clientes un martes flojo a unos 81 un sábado lleno, manteniendo la **misma anchura media** que antes. No se añade incertidumbre: se reparte donde corresponde.

Planificar con la predicción puntual dejaría al negocio expuesto la mitad de los días. El intervalo permite decidir con criterio: si el extremo alto exige dos camareros más, conviene tener a alguien localizable.

---

## 10. Del modelo al producto

La cifra sola no resuelve el problema del encargado. La aplicación la traduce en una decisión.

**Predicción → plantilla recomendada.** Se usa el ratio de clientes por empleado, con los mismos umbrales con los que se etiqueta la jornada en los datos de actividad:

| Ratio | Situación |
|---|---|
| Más de 24 clientes por empleado | Falta de personal |
| Entre 15 y 24 | Personal ocupado |
| Menos de 15 | Personal ocioso |

Se apunta al centro de la banda cómoda (18 clientes por empleado) para dejar margen en ambas direcciones. Un test comprueba que la recomendación **nunca deja el local en riesgo de falta de personal** en todo el rango de demanda observado.

**Explicación sin IA generativa.** La aplicación explica en lenguaje natural por qué sale ese número, pero lo hace con reglas sobre los valores reales que ha recibido el modelo, no con un modelo de lenguaje. Así la explicación no puede inventar una causa que no esté en los datos. Es una decisión deliberada: en un sistema de apoyo a la decisión, una explicación plausible pero falsa es peor que no dar ninguna.

---

## 11. Limitaciones

**El sesgo por crecimiento.** Descrito en la sección 9.1, junto con las cuatro correcciones probadas y el motivo por el que todas empeoran la decisión. No es una limitación pendiente de atacar: está atacada, medida y resuelta con vigilancia en lugar de con corrección.

**La previsión meteorológica tiene su propio error**, que se suma al del modelo. Y AEMET no publica milímetros previstos en la predicción municipal, solo probabilidad de precipitación: hay que estimar los milímetros como valor esperado.

**Un solo establecimiento.** El modelo aprende los patrones de un restaurante concreto. Aplicarlo a otro requeriría reentrenarlo con sus datos.

**No hay variables de promociones ni eventos locales.** Ya se identificaron como deseables en la Entrega 2 y siguen sin estar disponibles. Parte del error residual probablemente venga de ahí.

**Un mes sin datos de lluvia.** El pluviómetro de la estación 3182Y no dio dato del 19 de octubre al 20 de noviembre de 2022: 33 días seguidos, con las temperaturas correctas. Se rellenan con 0,0 mm, lo que equivale a afirmar que no llovió en un mes de otoño en Madrid. De esos 33 días, 26 entran en la capa gold y son el grueso de los 39 marcados como imputados.

Sobre estos datos el efecto es nulo, porque la actividad es sintética y se generó a partir de esta misma meteorología: el generador y el modelo vieron el mismo `prec = 0`, así que el error se cancela. **Con datos reales no se cancelaría.** Los días afectados están marcados en `meteo_imputada` para poder excluirlos, y el descargador vuelve a pedir esos días a AEMET por si ya los ha publicado.

---

## 12. Líneas de mejora

1. **Reentrenamiento automático.** `src/analysis/monitorizacion.py` ya dice *cuándo* hay que reentrenar; falta que lo lance solo. Es poco trabajo y cierra el ciclo del todo.

2. **Incorporar eventos locales.** Las fiestas patronales, los partidos y los conciertos del municipio son información pública y probablemente expliquen parte del error residual.

3. **Predicción por franjas.** Comida y cena tienen dinámicas distintas, y la plantilla se organiza por turnos. Requeriría datos con granularidad horaria.

4. **Un modelo no lineal para la temperatura.** La ceguera al calor extremo (sección 15) es estructural del modelo lineal. Un término cuadrático en `tmed`, o un spline, podría recuperar la curva sin renunciar a la interpretabilidad.

5. **Medir el error real de la previsión.** `src/extraction/archivar_prediccion.py` lleva meses guardando las predicciones de AEMET. Con un año de archivo, la sección 16 dejaría de ser un test de estrés y pasaría a ser una medición.

Dos líneas que estaban en esta lista y ya no lo están, porque se hicieron:

- *Componente de tendencia explícita.* Hecha y descartada con datos: sección 9.1.
- *Intervalos por regresión cuantílica.* Hechos y desplegados: sección 9.3.

---

## 13. Reproducibilidad

Todo el análisis se reproduce con:

```bash
python pipeline.py
```

El pipeline ejecuta 17 etapas en unos 30 segundos, desde la generación del calendario hasta el mockup del frontal, y funciona sin conexión: los datos descargados de AEMET están en el repositorio.

Todas las semillas aleatorias están fijadas. Las cifras de este documento, las de los notebooks y las que muestra la aplicación proceden de los mismos ficheros de resultados, de modo que no pueden divergir.

Los cuatro notebooks se entregan **ejecutados**, con sus salidas y sus catorce gráficos guardados en el propio `.ipynb`. Quien los abra ve los resultados sin tener que montar el entorno, y quien quiera comprobarlos puede volver a ejecutarlos: leen los mismos ficheros que el resto del proyecto.

El proyecto incluye **88 tests** (`pytest tests/`) que comprueban desde el cálculo de la Pascua hasta que la plantilla recomendada nunca deja el local en riesgo.

---

## 14. El sistema como herramienta de decisión

Todo lo anterior mide el error en clientes. Pero el encargado no decide clientes: decide personas. La pregunta que de verdad importa no es cuánto se equivoca el modelo, sino **si habría acertado con la plantilla**.

### 14.1. Cómo se define acertar

Con los mismos umbrales con los que se etiqueta la jornada en los datos de actividad:

```
plantilla mínima = ceil(clientes / 24)   por debajo, falta personal
plantilla máxima = ceil(clientes / 15)   por encima, sobra
```

No se exige clavar el número: se exige caer dentro de la banda cómoda.

### 14.2. Resultados sobre los 167 días de test

| Método | Plantilla adecuada | Falta de personal | Exceso de personal |
|---|---|---|---|
| Intuición (baseline) | 137 (82,0 %) | 17 (10,2 %) | 13 (7,8 %) |
| **Modelo** | **156 (93,4 %)** | **6 (3,6 %)** | **5 (3,0 %)** |
| Oráculo | 167 (100 %) | 0 | 0 |

El oráculo conoce de antemano los clientes exactos y marca el suelo teórico.

**El sistema reduce los días con falta de personal de 17 a 6.** Es el error caro, y es el que más se reduce.

### 14.3. Traducción a euros

| Parámetro | Valor | Origen |
|---|---|---|
| Coste de una jornada de camarero | 119 € | Convenio de hostelería de Madrid más cotización empresarial |
| Margen de contribución por cliente | 14,5 € | Ticket medio en España menos coste de materia prima |
| Demanda excedente que se pierde | 50 % | Supuesto |

| Método | Ventas perdidas | Salario de más | Total (7 meses) | Anual |
|---|---|---|---|---|
| Intuición | 2.269 € | 1.666 € | 3.935 € | 5.773 € |
| **Modelo** | **696 €** | **714 €** | **1.410 €** | **2.069 €** |

**Ahorro anual estimado: 3.704 €**, que es el **64 %** de la mejora alcanzable con predicción perfecta.

Los parámetros son referencias públicas del sector, no datos del establecimiento. Por eso el análisis se repitió con **doce combinaciones**, variando la fracción de demanda perdida entre 0,25 y 1,00 y el coste de la jornada entre 90 € y 150 €: **el modelo sale mejor en las doce**, con un ahorro de entre el 61 % y el 67 %.

La conclusión no depende de acertar con los supuestos económicos.

---

## 15. Validación contra el proceso generador

Un modelo puede tener buen error y haber aprendido la estructura equivocada. Con datos reales no hay forma de distinguirlo. Como aquí los datos de actividad están generados, sí se puede.

### 15.1. Método

Contrafactuales: se cambia una sola variable en todas las filas del test, se predice y se compara con el escenario de referencia. El cociente es el efecto que el modelo atribuye a esa variable, directamente comparable con el que aplica el generador.

Los valores reales **se importan del propio generador**, no se escriben a mano.

### 15.2. Resultados

| Bloque | Desvío medio | Valoración |
|---|---|---|
| Festivo | 1,8 % | Lo recupera bien |
| Mes | 6,7 % | Lo recupera bien |
| Día de la semana | 9,9 % | Lo recupera bien |
| Lluvia (entre semana) | 12,5 % | Aproximado |
| Lluvia (fin de semana) | 21,6 % | Aproximado |
| Temperatura | 21,9 % | Aproximado |

**Correlación entre efecto real y recuperado: 0,971.** El modelo ha aprendido la estructura correcta del problema.

### 15.3. Dónde falla, y por qué

| Temperatura media | Efecto real | Regresión lineal | Hist Gradient Boosting |
|---|---|---|---|
| 21 °C | 1,00 | 1,00 | 1,00 |
| 33 °C | 0,77 | 1,07 | — |
| 38 °C | **0,65** | **1,11** | **0,88** |

A 38 °C la demanda real cae un 35 %. **La regresión lineal predice que sube un 11 %.**

La causa es estructural: el efecto de la temperatura sube y luego baja, y una recta tiene una sola pendiente. O capta el frío o capta el calor.

Esto matiza la elección de modelo del apartado 7.3: **la regresión lineal gana porque los días de calor extremo son pocos, no porque los modele bien.** Es una conclusión que no se puede alcanzar mirando solo el MAE.

---

## 16. Robustez frente al error de la previsión

El modelo se entrena y evalúa con la meteorología **observada**, pero se usa con la **previsión**, que se equivoca. El MAE publicado corresponde por tanto a un escenario imposible.

AEMET no publica un archivo histórico de sus predicciones, así que el error real no se puede medir todavía. En lugar de inventar una cifra, se degrada la meteorología a propósito en un rango amplio y se mide la respuesta.

| Antelación | MAE | Incremento | Días con plantilla correcta |
|---|---|---|---|
| Meteorología observada | 16,07 | — | 157,0 |
| 1 día | 17,00 | +5,8 % | 154,0 |
| 3 días | 17,99 | +11,9 % | 151,1 |
| 5 días | 18,97 | +18,0 % | 149,0 |
| 7 días | 19,92 | +23,9 % | 146,5 |

**El sistema aguanta bien la previsión a corto plazo.** La recomendación operativa es no planificar con más de tres días de antelación, lo que encaja con el uso real: la plantilla del fin de semana se cierra el jueves.

Se ha implementado `src/extraction/archivar_prediccion.py`, que guarda cada día la previsión publicada. Tras unos meses de ejecución permitirá sustituir estos supuestos por la medición real.
