# PI_M5_V1 - MLOps Pipeline (Monitoreo y Data Drift)

Proyecto de MLOps para predicción de riesgo crediticio con:

- API de inferencia en FastAPI
- App de monitoreo en Streamlit
- Detección de data drift con PSI (Population Stability Index)
- Predicciones batch con exportación de resultados

## Arquitectura actual

```text
mlops_pipeline/
├── Dockerfile
├── requirements.txt
├── README.md
└── src/
    ├── model_deploy.py
    ├── model_monitoring.py
    ├── model_training_evaluation.py
    ├── ft_engineering.py
    ├── cargar_datos.py
    ├── models/
    │   ├── RandomForestClassifier_optuna.pkl
    │   └── feature_names.pkl
    └── predicciones/ (generadas al hacer predicciones por lotes en Streamlit)
        └── predicciones_batch_YYYYMMDD_HHMMSS.csv
```

## Componentes

### 1) API de inferencia (`src/model_deploy.py`)

Endpoints disponibles:

- `POST /predict`: predicción individual
- `POST /predict_batch`: predicción por lote

La API carga:

- `src/models/RandomForestClassifier_optuna.pkl`
- `src/models/feature_names.pkl`

## 2) Monitoreo (`src/model_monitoring.py`)

La app Streamlit tiene 4 tabs:

- **Graficas**: distribución de predicciones y comparación con referencia
- **Data Drift**: PSI por variable, alertas y evolución temporal
- **Logs**: vista tabular + descarga CSV desde la UI
- **Predicciones por Lotes**: carga CSV, consulta API batch y exporta resultados

### Comportamiento importante del monitoreo

- El drift se calcula con split base del dataset:
  - **Referencia:** 80%
  - **Actual:** 20%
- Los archivos generados por batch en `src/predicciones/` **no** se usan para cálculo de drift.
- Cada ejecución batch crea un archivo en:
  - `src/predicciones/predicciones_batch_YYYYMMDD_HHMMSS.csv`

## 3) Resumen de archivos clave de modelado

### `src/ft_engineering.py`

Este módulo concentra todo el preprocesamiento y devuelve los datasets listos para entrenar/evaluar:

- Carga y limpia datos (nulos, outliers y consistencia de tipos).
- Crea features derivadas de negocio (grupo de edad, variables temporales, total de créditos, etc.).
- Evita leakage eliminando variables que no deben entrar al modelo.
- Ordena temporalmente y aplica split 80/20 sin shuffle.
- Aplica pipeline de Feature-engine (imputación + encoding) y devuelve:
  - `X_train_processed_fe`, `X_test_processed_fe`, `y_train`, `y_test`.

### `src/model_training_evaluation.py`

Este módulo entrena, compara y optimiza modelos de clasificación:

- Define tres candidatos: RandomForest, XGBoost y CatBoost.
- Evalúa con `TimeSeriesSplit` y métricas de clasificación, priorizando clase 0 (`recall_0` y `f1_0`).
- Selecciona el mejor modelo base con criterio robusto (`mean - std`).
- Ejecuta optimización de hiperparámetros con Optuna sobre el mejor candidato.
- Entrena el modelo final, guarda artefactos en `src/models/`:
  - modelo `*_optuna.pkl`
  - `feature_names.pkl` (orden de columnas esperado por la API).

## 4) Data Drift (PSI)

Interpretación usada en la app:

- `PSI < 0.10` → 🟢 Bajo
- `0.10 <= PSI <= 0.25` → 🟡 Moderado
- `PSI > 0.25` → 🔴 Alto

Variables temporales excluidas del análisis de drift:

- `mes_prestamo`
- `anio_prestamo`
- `dia_semana_prestamo`
- `fin_de_mes`

## Instalación y ejecución local

### 1) Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2) Entrenar (si necesitás regenerar modelos)

```bash
cd src
python model_training_evaluation.py
```

### 3) Levantar API

```bash
cd src
uvicorn model_deploy:app --reload
```

API docs:

- `http://localhost:8000/docs`

### 4) Levantar Streamlit

```bash
cd src
streamlit run model_monitoring.py
```

UI:

- `http://localhost:8501`

## Ejecución con Docker (solo API)

### Build de imagen

```bash
docker build -t mlops-api .
```

### Run del contenedor

```bash
docker run -d --name mlops-api-container -p 8000:8000 mlops-api
```

### Verificar

- API: `http://localhost:8000/docs`

> Streamlit sigue ejecutándose localmente (host) en `8501` y consume la API en `localhost:8000`. Solamente consume el endpoint de predicciones por lotes `/predict_batch`

## Deploy en nube (Streamlit público)

### ¿Se puede en Vercel?

Para este proyecto, **no es la mejor opción** para publicar Streamlit directamente.

- Vercel está optimizado para frontends y funciones serverless.
- Streamlit necesita un proceso Python web vivo (estado de sesión + websocket), no una función efímera.
- Resultado típico en Vercel: timeouts, reinicios o comportamiento inestable.

### Opción recomendada: Render (API + Streamlit)

Con Render podés desplegar ambos servicios y dejarlos públicos.

#### 1) Subir repo a GitHub

Asegurate de tener esta estructura en el repo:

- `mlops_pipeline/src/model_deploy.py`
- `mlops_pipeline/src/model_monitoring.py`
- `mlops_pipeline/requirements.txt`

#### 2) Crear servicio API (Web Service)

En Render:

- **Runtime:** Python
- **Root Directory:** `mlops_pipeline`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn src.model_deploy:app --host 0.0.0.0 --port $PORT`

Al deployar obtendrás una URL tipo:

- `https://tu-api.onrender.com`

Probá:

- `https://tu-api.onrender.com/docs`

#### 3) Crear servicio Streamlit (Web Service)

Nuevo servicio en Render (mismo repo):

- **Runtime:** Python
- **Root Directory:** `mlops_pipeline`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run src/model_monitoring.py --server.port $PORT --server.address 0.0.0.0`

Variables de entorno en este servicio:

- `API_BASE_URL=https://tu-api.onrender.com`

Con esto, la pestaña **Predicciones por Lotes** dejará de usar localhost y consumirá tu API pública.

### Opción rápida: Streamlit Community Cloud

Si querés ir más rápido para demo:

1. Crear app en Streamlit Cloud apuntando al repo.
2. Main file path: `mlops_pipeline/src/model_monitoring.py`.
3. Agregar `API_BASE_URL` en Secrets/Variables.
4. Publicar.

### Si querés usar Vercel igual

Patrón recomendado:

- Deploy del frontend en Vercel (Next.js/React).
- API ML en Render/Railway/Fly.io.
- No deployar Streamlit en Vercel para producción.

## Flujo recomendado de uso

1. Levantar API (`uvicorn` o Docker)
2. Levantar Streamlit
3. Visualizar tabs de **Graficas** y **Data Drift**
4. Ir a **Predicciones por Lotes**
5. Cargar un CSV con columnas esperadas por el modelo
6. Ejecutar predicciones
7. Descargar resultados desde la UI

## Notas de mantenimiento

- Si cambiás features del modelo, regenerá `feature_names.pkl` y el `.pkl` del modelo.
- Si cambian puertos/host de API, actualizá las URLs de consumo en `model_monitoring.py`.
- La carpeta `src/predicciones/` actúa como salida de corridas batch.
