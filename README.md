# UAM_TFM

Repositorio correspondiente al Trabajo de Fin de Máster (TFM), centrado en la evaluación del modelo **TimesFM** y su comparación con otros modelos de predicción de series temporales. El proyecto incluye tanto experimentos de *fine-tuning* sobre TimesFM como pruebas de inferencia y evaluación utilizando versiones preentrenadas del modelo (*zero-shot*).

La organización del repositorio está dividida en dos bloques principales, cada uno con sus propios notebooks y módulos auxiliares.

---

# Estructura del repositorio

```text
TFM/
│
├── finetune_timesfm_folder/
│   ├── utils_folder/
│   │   ├── data_util.py
│   │   ├── comp_util.py
│   │   ├── model_util.py
│   │   └── plot_util.py
│   ├── finetune_ex.ipynb
│   └── requirements.txt
│
└── timesfm_25_folder/
    ├── utils_folder/
    │   ├── data_util.py
    │   ├── timesfm_25_util.py
    │   └── plot_util.py
    ├── test_model.ipynb
    └── requirements.txt
```

---

# 1. Fine-tuning de TimesFM

La carpeta `finetune_timesfm_folder` contiene todo el código relacionado con el ajuste fino del modelo TimesFM y la evaluación comparativa frente a otros modelos de forecasting.

## Archivos principales

### `finetune_ex.ipynb`

Notebook principal del bloque de TimesFM 2.0.

En este notebook se realizan las distintas fases experimentales:

- Carga y preparación de los conjuntos de datos.
- Configuración del proceso de entrenamiento.
- Fine-tuning de TimesFM.
- Entrenamiento o ejecución de modelos de comparación.
- Generación de predicciones.
- Cálculo de métricas de evaluación.
- Comparación de resultados entre modelos.
- Visualización y análisis de resultados.

Este notebook actúa como punto central desde el que se ejecutan todos los experimentos relacionados con el ajuste fino.

### `requirements.txt`

Contiene las dependencias necesarias para reproducir los experimentos de esta sección.

---

## Carpeta `utils_folder`

Agrupa las funciones auxiliares utilizadas por el notebook principal.

### `data_util.py`

Módulo encargado de la generación, carga y preparación de los datos utilizados en los experimentos.

Sus principales funcionalidades son:

- Generación de series temporales sintéticas basadas en funciones seno con diferentes niveles de complejidad.
- Descarga y preparación de datos financieros reales mediante Yahoo Finance.
- Carga y procesamiento de datasets de demanda energética almacenados en formato Parquet.
- Detección automática de periodos estacionales mediante la Función de Autocorrelación (ACF).
- Aplicación de técnicas de descomposición temporal:
-   Diferenciación de tendencia.
-   Diferenciación estacional.
-   Eliminación de componentes estacionales mediante STL.
-   Eliminación conjunta de tendencia y estacionalidad.
- Normalización de datos mediante StandardScaler.
- Construcción de ventanas deslizantes para entrenamiento y evaluación.
- Creación de datasets compatibles con la arquitectura TimesFM mediante la clase TimeSeriesDataset.

Este módulo centraliza todo el pipeline de preparación de datos utilizado por TimesFM y por los modelos de comparación.

### `model_util.py`

Módulo principal de entrenamiento, inferencia y evaluación de modelos.

Incluye funcionalidades para:

Descarga y carga automática del modelo preentrenado TimesFM desde Hugging Face.
Construcción de modelos TimesFM para evaluación zero-shot y fine-tuning.
Ejecución del proceso de ajuste fino mediante TimesFMFinetuner.
Evaluación de modelos utilizando RMSE como métrica principal.
Reconstrucción de predicciones escaladas y reintegración de componentes eliminadas durante la descomposición.
Comparación experimental entre:
TimesFM Zero-Shot.
TimesFM Fine-Tuned.
ARIMA.
Persistence Forecast.
Dummy Regressor.
LSTM Multi-Output.
LSTM Multi-Model.
LSTM Autorregresivo.
Medición de tiempos de entrenamiento e inferencia.
Evaluación de estabilidad de modelos LSTM mediante múltiples ejecuciones.
Generación de tablas resumen con métricas comparativas.

La función principal del módulo es compare_performance(), que ejecuta de forma automatizada todo el proceso experimental, desde la carga de datos hasta la comparación final de resultados.

### `comp_util.py`

Módulo destinado a la evaluación y comparación de modelos.

Incluye utilidades para:

- Cálculo de métricas de error.
- Comparación cuantitativa entre modelos.
- Organización y almacenamiento de resultados.
- Generación de tablas resumen para el análisis experimental.

Este archivo constituye el núcleo de la comparación entre TimesFM y los modelos de referencia utilizados en el estudio.

### `plot_util.py`

Módulo dedicado a la representación gráfica de resultados.

Permite generar:

- Gráficos de predicciones frente a valores reales.
- Comparaciones visuales entre modelos.
- Figuras utilizadas para el análisis experimental.

---

# 2. Evaluación con TimesFM 2.5

La carpeta `timesfm_25_folder` contiene los experimentos realizados utilizando la versión TimesFM 2.5 sin procesos adicionales de *fine-tuning*.

Su objetivo principal es analizar el rendimiento del modelo y compararlo con el resto de alternativas consideradas en el trabajo.

## Archivos principales

### `test_model.ipynb`

Notebook principal de experimentación.

En él se realizan:

- Carga de datasets.
- Inicialización de TimesFM 2.5.
- Ejecución de predicciones.
- Evaluación mediante métricas de forecasting.
- Generación de resultados y figuras para el análisis.

### `requirements.txt`

Dependencias necesarias para ejecutar los experimentos de esta carpeta.

---

## Carpeta `utils_folder`

Contiene funciones auxiliares específicas para la evaluación de TimesFM 2.5.

### `data_util.py`

Funciones para:

- Lectura de datos.
- Preprocesamiento.
- Construcción de conjuntos de evaluación.
- Adaptación de datos al formato requerido por los modelos.

### `timesfm_25_util.py`

Módulo específico para la interacción con TimesFM 2.5.

Incluye funcionalidades relacionadas con:

- Carga del modelo.
- Configuración de parámetros de inferencia.
- Generación de predicciones.
- Gestión de entradas y salidas del modelo.

### `plot_util.py`

Herramientas para la visualización de resultados experimentales.

Permite representar:

- Predicciones frente a observaciones reales.
- Comparativas entre modelos.

---

# Flujo general de trabajo

El flujo seguido en el proyecto puede resumirse en los siguientes pasos:

1. Carga y preparación de los datasets.
2. Ejecución de TimesFM (con o sin fine-tuning según el experimento).
3. Ejecución de los modelos de comparación.
4. Obtención de predicciones para cada modelo.
5. Cálculo de métricas de evaluación.
6. Comparación cuantitativa y visual de resultados.
7. Análisis de rendimiento y elaboración de conclusiones.

---

# Objetivo del repositorio

El objetivo principal de este repositorio es proporcionar un entorno reproducible para evaluar el rendimiento de **TimesFM** en tareas de predicción de series temporales y comparar sus resultados con distintos modelos de referencia mediante métricas y análisis visuales homogéneos.
