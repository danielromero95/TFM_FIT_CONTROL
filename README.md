# Gym Performance Analysis

**Repositorio:** `gym-performance-analysis`  
**Autor:** Daniel Romero  
**Última actualización:** 4 de junio de 2025

---

## Descripción

Este proyecto tiene como objetivo construir un sistema de análisis de rendimiento en gimnasio basado en visión por computador y datos de wearables. A partir de vídeos de ejercicios de fuerza (squat, bench press, deadlift) y datos de acelerómetro/pulso, el sistema debe:

1. Detectar el tipo de ejercicio.
2. Contar repeticiones de manera automática.
3. Calcular ángulos articulares clave (rodilla, cadera, hombro).
4. Detectar posibles errores de técnica con ≥ 90 % de precisión.
5. Sincronizar las señales de vídeo con datos de wearables.
6. Ofrecer una demo web mínima para visualizar resultados.

El flujo general contempla:
- **Preprocesado de vídeo:** extracción de fotogramas, redimensionado, normalización, filtrado (Gauss, CLAHE), recorte de ROI.
- **Estimación de pose:** MediaPipe Pose para extraer landmarks y calcular ángulos, velocidades angulares, simetría, etc.
- **Procesamiento de wearables:** interpolación de acelerómetro y pulso para alinearlo con cada fotograma.
- **Construcción de dataset:** combinación de un dataset público (Exercise-Recognition/Kaggle) y de grabaciones propias (≥ 1 000 vídeos) etiquetadas “correcto/incorrecto”.
- **Modelado:**  
  - Conteo de repeticiones mediante detección de picos en las series de ángulos.  
  - Detección de fallos (baseline con Random Forest/SVM, versión avanzada con DL ligera en Keras/TensorFlow o PyTorch).  
- **Validación:** cross‐validation k=5, métricas (precision, recall, F1); pruebas con usuarios (5–10 voluntarios), encuesta SUS.
- **Despliegue:** contenedorización con Docker, API REST para recibir vídeo+CSV de sensores y devolver resultados en JSON, demo en Streamlit (o React).
- **MLOps:** uso de MLflow para tracking de experimentos y GitHub Actions para CI/CD.

Este repositorio contiene la estructura base para comenzar el desarrollo de cada uno de los módulos de forma modular y rastreable mediante Git.

---

## Estructura de carpetas

```plaintext
gym-performance-analysis/
│
├─ .venv/                       ← Entorno virtual (no se sube a remoto)
│
├─ data/                        ← Carpeta para datos
│   ├─ raw/                     ← Vídeos originales (Exercise-Recognition + propios)
│   │   └─ …                     ← (copiar aquí los clips en crudo)
│   └─ processed/               ← Datos procesados (fotogramas, CSVs intermedios)
│       └─ …
│
├─ notebooks/                   ← Jupyter Notebooks de exploración y prototipado
│   ├─ 01_exploracion_dataset.ipynb
│   └─ …
│
├─ src/                         ← Código fuente organizado en módulos
│   ├─ preprocessing/           ← Extracción de fotogramas, redimensionado, filtros
│   │   └─ frame_extraction.py
│   │   └─ image_preprocessing.py
│   │   └─ …                     ← (otros scripts de preprocesado)
│   │
│   ├─ pose_estimation/         ← Llamadas a MediaPipe, filtrado de keypoints, cálculo de ángulos
│   │   └─ pose_utils.py
│   │   └─ …                     ← (scripts de cálculo de vectores/ángulos)
│   │
│   ├─ sensor_preprocessing/     ← Lectura, alineación e interpolación de datos de wearables
│   │   └─ sensor_utils.py
│   │   └─ …                     ← (otros scripts de preprocesado de sensores)
│   │
│   ├─ modeling/                ← Definición y entrenamiento de modelos
│   │   ├─ count_reps.py        ← Algoritmo de conteo de repeticiones (basado en picos)
│   │   ├─ fault_detection.py   ← Baseline (Random Forest/SVM) para detección de fallos
│   │   └─ dl_model.py          ← Red ligera (MLP/LSTM) para detección de fallos avanzados
│   │
│   └─ deployment/              ← Código de despliegue y API REST
│       └─ api.py               ← Endpoint `/analyze` que procesa vídeo + CSV sensor
│       └─ docker/              ← Dockerfile y scripts de construcción de imagen
│
├─ app_demo/                    ← Demo en Streamlit (o React) para cargar vídeo + CSV
│   └─ demo_streamlit.py        ← Ejemplo mínimo de interface que muestra métricas
│   └─ requirements_streamlit.txt
│
├─ docs/                        ← Documentación adicional
│   ├─ anteproyec
