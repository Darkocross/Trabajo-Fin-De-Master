# Entrega 2 · Selección de la idea de proyecto y análisis de los datos necesarios

**Por Juan Francisco Morales Olivo**

## 1. Idea seleccionada

### Idea del proyecto

**Predicción de la demanda en establecimientos de hostelería a partir de variables meteorológicas.**

### Problema que pretende resolver

Uno de los principales retos del sector de la hostelería es anticipar la afluencia de clientes para planificar correctamente los recursos necesarios en cada jornada.

En muchos establecimientos, decisiones como la contratación de personal, la compra de productos o la preparación del servicio se toman principalmente en función de la experiencia del responsable del negocio. Sin embargo, existen factores externos, como la temperatura, la lluvia, el viento o el estado del tiempo, que pueden influir de forma importante en el comportamiento de los clientes.

Una previsión poco precisa puede traducirse en un exceso de personal y de costes o, por el contrario, en falta de recursos para atender la demanda, afectando tanto a la rentabilidad del negocio como a la experiencia del cliente.

### Solución propuesta

El proyecto consiste en desarrollar un sistema de análisis de datos que estudie la relación entre las condiciones meteorológicas y la demanda en establecimientos de hostelería.

Para ello se utilizarán datos históricos de ventas o de afluencia de clientes junto con información meteorológica obtenida de fuentes públicas. En una primera fase se realizará un análisis exploratorio para identificar patrones y relaciones entre ambas variables.

Posteriormente se entrenará un modelo predictivo capaz de estimar la demanda esperada para una fecha determinada utilizando variables como:

- Temperatura máxima.
- Temperatura mínima.
- Precipitación.
- Humedad relativa.
- Estado del tiempo.

El objetivo es disponer de una herramienta que sirva de apoyo en la planificación diaria del negocio.

### MVP del proyecto

El producto mínimo viable consistirá en un dashboard interactivo que permita:

- Visualizar la evolución histórica de la demanda junto con las variables meteorológicas.
- Consultar la previsión meteorológica de un día concreto.
- Obtener una estimación del número de clientes o del volumen de ventas esperado.
- Mostrar algunas métricas básicas sobre la calidad de la predicción.

De esta forma, el proyecto combinará análisis exploratorio, visualización de datos y un modelo predictivo funcional.

---

# 2. Datos necesarios

Para el desarrollo del proyecto serán necesarios dos conjuntos principales de información:

- Datos de actividad del negocio.
- Datos meteorológicos.

## Datos de actividad del negocio

### Variables necesarias

- Fecha.
- Hora (si está disponible).
- Número de clientes.
- Número de tickets o transacciones.
- Importe total de ventas.
- Identificador del establecimiento.
- Día de la semana.
- Indicador de festivo.

## Datos meteorológicos

### Variables necesarias

- Temperatura máxima.
- Temperatura mínima.
- Temperatura media.
- Precipitación acumulada.
- Humedad relativa.
- Velocidad del viento.
- Presión atmosférica.
- Estado del cielo o descripción meteorológica.
- Horas de sol (si estuvieran disponibles).

### Granularidad

La granularidad más adecuada sería disponer de un registro por **día y establecimiento**, ya que permite relacionar directamente la demanda diaria con las condiciones meteorológicas correspondientes.

Si fuese posible obtener información horaria, podría ampliarse el análisis para realizar predicciones más precisas, aunque no es un requisito para el MVP.

### Profundidad histórica

Sería recomendable disponer de entre **dos y cinco años** de información histórica, incluyendo todas las estaciones del año, periodos vacacionales y festivos.

Esta amplitud temporal permitiría capturar patrones estacionales y posibles variaciones relacionadas con el clima.

### Volumen aproximado

Tomando como ejemplo una cadena ficticia de cinco establecimientos durante tres años:

```text
365 días × 3 años × 5 establecimientos ≈ 5.500 registros
```

Este volumen resulta suficiente para realizar análisis estadísticos y entrenar modelos predictivos de complejidad media.

### Datos imprescindibles

- Fecha.
- Número de clientes o volumen de ventas.
- Temperatura.
- Precipitación.
- Identificador del establecimiento.

### Datos deseables

- Humedad.
- Velocidad del viento.
- Horas de sol.
- Información sobre promociones.
- Eventos locales.
- Datos horarios.
- Calendario de festivos nacionales y regionales.

---

# 3. Fuentes de datos previstas

## Fuente principal: AEMET

La principal fuente de datos meteorológicos será la Agencia Estatal de Meteorología (AEMET).

**Sitio web**

https://www.aemet.es

**Portal OpenData**

https://opendata.aemet.es

### Características

- Datos abiertos.
- Acceso mediante API.
- Disponibilidad de históricos meteorológicos.
- Información oficial mantenida por un organismo público.

### Formato esperado

- JSON mediante API.
- CSV tras el proceso de transformación.

---

## Fuente alternativa: Open-Meteo

**Sitio web**

https://open-meteo.com

### Características

- API gratuita.
- Acceso sencillo.
- Datos históricos y previsiones meteorológicas.
- Documentación completa.

### Formato esperado

- JSON.

---

## Datos de actividad del negocio

Como los datos reales de ventas suelen ser privados, se contemplan dos alternativas.

### Opción 1. Datos simulados

Generar un conjunto de datos sintético que reproduzca patrones realistas de consumo en función de la meteorología.

**Ventajas**

- Disponibilidad inmediata.
- Sin problemas legales ni de privacidad.
- Control total sobre la calidad de los datos.

### Opción 2. Datos abiertos relacionados

Buscar conjuntos de datos públicos de ventas, afluencia o consumo en plataformas como:

- Kaggle.
- Data.gov.
- Portal Europeo de Datos Abiertos.

**Posibles formatos**

- CSV.
- Excel.
- Bases de datos públicas.

### Riesgos identificados

- Dificultad para obtener datos reales de ventas.
- Históricos incompletos.
- Datos meteorológicos faltantes en algunas fechas.
- Cambios en las APIs o en sus límites de acceso.
- Necesidad de generar datos sintéticos para completar el caso de estudio.

---

# 4. Consideraciones sobre privacidad y protección de datos

El proyecto está planteado para minimizar cualquier riesgo relacionado con la protección de datos.

## Datos personales

No está previsto utilizar información como:

- Nombres de clientes.
- Direcciones.
- Correos electrónicos.
- Teléfonos.
- Cualquier dato que permita identificar a una persona.

## Anonimización

En caso de trabajar con datos reales, únicamente se utilizarían datos agregados por fecha y establecimiento, evitando cualquier información individual de los clientes.

## Uso en un entorno académico

Los datos meteorológicos son públicos y no contienen información personal.

Los datos de actividad del negocio serán agregados o simulados, por lo que el proyecto puede desarrollarse sin comprometer la privacidad de ninguna persona.

## Riesgos éticos o legales

Los riesgos asociados son reducidos porque:

- No se elaboran perfiles individuales.
- No se toman decisiones automatizadas sobre personas.
- No se utilizan categorías especiales de datos protegidos.

## Datos que se evitarán

No se utilizarán:

- Datos individuales de clientes.
- Información identificable sobre pagos.
- Información personal de empleados.

---

# 5. Viabilidad inicial del proyecto

La viabilidad general del proyecto es alta.

Los datos meteorológicos pueden obtenerse fácilmente a través de fuentes abiertas como AEMET o Open-Meteo, que ofrecen históricos suficientes para realizar análisis temporales.

La principal dificultad está en la obtención de datos reales de ventas o afluencia de clientes, ya que este tipo de información suele pertenecer a empresas privadas.

Aun así, el proyecto puede desarrollarse de forma realista dentro del contexto del máster, ya que combina análisis exploratorio, visualización de datos y modelado predictivo utilizando un volumen de información perfectamente manejable.

En caso de no poder acceder a datos reales de negocio, se utilizarán conjuntos de datos públicos similares o se generará un dataset sintético basado en patrones observados y datos meteorológicos reales. Esta alternativa permitirá desarrollar todas las fases del proyecto y validar la viabilidad técnica de la propuesta.