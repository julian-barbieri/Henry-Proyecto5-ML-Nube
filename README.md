# MLOps Pipeline — Predicción de Riesgo Crediticio

Pipeline de MLOps end-to-end para predicción de incumplimiento de pago en préstamos, con API de inferencia, monitoreo de data drift y predicciones por lotes.

**Demo en producción:**
- 🔗 API REST: [henry-proyecto5-ml-api.onrender.com](https://henry-proyecto5-ml-api.onrender.com/docs)
- 📊 App Streamlit: [henry-proyecto5-ml-nube-streamlit.onrender.com](https://henry-proyecto5-ml-nube-streamlit.onrender.com)

---

## Arquitectura

```
Base_de_datos.xlsx
       │
       ▼
cargar_datos.py          ← Carga y parseo del Excel
       │
       ▼
ft_engineering.py        ← Limpieza, features y preprocesamiento
       │
       ▼
model_training_evaluation.py  ← Entrenamiento, comparación y optimización (Optuna)
       │
       ▼
src/models/
  ├── RandomForestClassifier_optuna.pkl
  └── feature_names.pkl
       │
       ▼
model_deploy.py (FastAPI) ← Inferencia individual y por lotes
       │
       ▼
model_monitoring.py (Streamlit) ← Monitoreo, drift y predicciones batch
```

---

## Estructura del repositorio

```
Henry-Proyecto5-ML-Nube/
├── Dockerfile                        # Imagen Docker para la API
├── requirements.txt                  # Dependencias completas (entrenamiento + Streamlit)
├── requirements_api.txt              # Dependencias mínimas para la API
├── Base_de_datos.xlsx                # Dataset fuente
├── ejemplo_batch_predictions.csv     # CSV de ejemplo para predicciones por lotes
├── .dockerignore
└── src/
    ├── cargar_datos.py               # Carga del Excel
    ├── ft_engineering.py             # Feature engineering y preprocesamiento
    ├── model_training_evaluation.py  # Entrenamiento y optimización de modelos
    ├── model_deploy.py               # API FastAPI
    ├── model_monitoring.py           # App Streamlit de monitoreo
    └── models/
        ├── RandomForestClassifier_optuna.pkl
        └── feature_names.pkl
```

---

## Componentes

### 1. Carga de datos (`src/cargar_datos.py`)

Lee `Base_de_datos.xlsx` desde la raíz del proyecto y parsea la columna `fecha_prestamo` como datetime. Devuelve un DataFrame listo para el pipeline de features.

---

### 2. Feature Engineering (`src/ft_engineering.py`)

Módulo central de preprocesamiento. Recibe los datos crudos y devuelve cuatro objetos listos para modelado: `X_train`, `X_test`, `y_train`, `y_test`.

**Limpieza aplicada:**
- Nulos en `tendencia_ingresos` → `"Sin_informacion"`
- Nulos en `promedio_ingresos_datacredito` y `puntaje_datacredito` → mediana
- Nulos en columnas de saldo → `0`
- Eliminación de fechas futuras (préstamos históricos)
- Outliers: edad > 90, plazo > 60 meses, salario y otros préstamos > percentil 99

**Features construidas:**
- `grupoEdad` — segmentación etaria (Joven / Adulto / Mayor)
- `es_independiente` — binaria desde `tipo_laboral`
- `anio_prestamo`, `mes_prestamo`, `dia_semana_prestamo` — descomposición temporal
- `fin_de_mes` — indicador binario de riesgo temporal
- `total_creditos` — suma de créditos activos por sector
- `ratio_mora_saldo` — proporción de mora sobre saldo total (excluida del modelo por leakage)

**Preprocesamiento con Feature-engine:**
- Imputación por mediana en numéricas
- Imputación por moda en categóricas y ordinales
- Encoding ordinal para `grupoEdad`
- One-hot encoding para `tipo_credito` y `tendencia_ingresos`

**Split temporal:** 80/20 sin shuffle, ordenado por `fecha_prestamo` para respetar la naturaleza secuencial de los datos y evitar leakage.

**Variables excluidas por leakage:** `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`, `ratio_mora_saldo`, `puntaje`.

---

### 3. Entrenamiento y Evaluación (`src/model_training_evaluation.py`)

Compara tres modelos candidatos y optimiza el mejor con Optuna.

**Modelos evaluados:**
- `RandomForestClassifier`
- `XGBClassifier`
- `CatBoostClassifier`

**Validación:** `TimeSeriesSplit` con 5 folds para respetar el orden temporal.

**Métrica de selección:** `recall_0` (recall de la clase 0 — préstamos en mora), usando el criterio robusto `mean - std` para penalizar modelos inestables.

**Optimización:** Optuna con 50 trials sobre el modelo ganador, maximizando `recall_0`.

**Artefactos generados en `src/models/`:**
- `{NombreModelo}_optuna.pkl` — modelo entrenado con mejores hiperparámetros
- `feature_names.pkl` — orden exacto de columnas usado en el entrenamiento

**Para reentrenar:**
```bash
cd src
python model_training_evaluation.py
```

---

### 4. API de Inferencia (`src/model_deploy.py`)

API REST construida con FastAPI. Carga el modelo y el orden de features al iniciar.

**Endpoints:**

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio |
| `POST` | `/predict` | Predicción individual |
| `POST` | `/predict_batch` | Predicción por lote (lista de registros) |

**Respuesta de `/predict` y `/predict_batch`:**
```json
{"Predicted_default": 0}
```
Donde `0` = pagó a tiempo, `1` = incumplimiento.

**Swagger interactivo:** [henry-proyecto5-ml-api.onrender.com/docs](https://henry-proyecto5-ml-api.onrender.com/docs)

---

### 5. App de Monitoreo (`src/model_monitoring.py`)

App Streamlit con 4 pestañas:

**Tab 1 — Gráficas:** distribución de predicciones del conjunto de validación y comparación de medias por variable contra el conjunto de referencia.

**Tab 2 — Data Drift (PSI):** calcula el Population Stability Index para cada feature, genera alertas automáticas y muestra la evolución temporal del drift por ventana configurable.

Interpretación del PSI:
- `PSI < 0.10` → 🟢 Estable
- `0.10 ≤ PSI ≤ 0.25` → 🟡 Moderado
- `PSI > 0.25` → 🔴 Alto — reentrenamiento recomendado

Variables excluidas del análisis de drift: `mes_prestamo`, `anio_prestamo`, `dia_semana_prestamo`, `fin_de_mes` (cambian naturalmente con el tiempo).

**Tab 3 — Logs:** tabla de predicciones del set de validación con descarga en CSV.

**Tab 4 — Predicciones por Lotes:** carga un CSV, lo envía a `/predict_batch` en chunks de 50 registros con reintentos automáticos ante errores 429, y permite descargar los resultados. La columna `Pago_atiempo` aparece primera en el resultado.

---

## Instalación local

### Requisitos
- Python 3.10+
- `Base_de_datos.xlsx` en la raíz del proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/julian-barbieri/Henry-Proyecto5-ML-Nube.git
cd Henry-Proyecto5-ML-Nube
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Reentrenar el modelo (opcional, si no tenés los .pkl)
```bash
cd src
python model_training_evaluation.py
```
Esto genera `src/models/RandomForestClassifier_optuna.pkl` y `src/models/feature_names.pkl`.

### 4. Levantar la API
```bash
cd src
uvicorn model_deploy:app --reload
```
Disponible en `http://localhost:8000/docs`

### 5. Levantar la app de monitoreo
En una terminal separada:
```bash
cd src
streamlit run model_monitoring.py
```
Disponible en `http://localhost:8501`

---

## Ejecución con Docker (solo API)

### Build
```bash
docker build -t mlops-api .
```

### Run
```bash
docker run -d --name mlops-api-container -p 8000:8000 mlops-api
```

Verificar: `http://localhost:8000/docs`

> El Dockerfile usa `requirements_api.txt` (dependencias mínimas) para mantener la imagen liviana. Streamlit se ejecuta por separado.

---

## Deploy en Render

### API (Docker)

| Campo | Valor |
|-------|-------|
| Runtime | Docker |
| Root Directory | *(vacío)* |
| Dockerfile Path | `./Dockerfile` |
| Instance Type | Free |

### Streamlit (Python)

| Campo | Valor |
|-------|-------|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run src/model_monitoring.py --server.port $PORT --server.address 0.0.0.0` |

**Variable de entorno requerida en el servicio Streamlit:**
```
API_BASE_URL = https://henry-proyecto5-ml-api.onrender.com
```

---

## Predicciones por lotes — Formato del CSV

El CSV debe contener exactamente estas columnas (ver `ejemplo_batch_predictions.csv`):

`capital_prestado`, `plazo_meses`, `edad_cliente`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `puntaje_datacredito`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal`, `promedio_ingresos_datacredito`, `grupoEdad`, `es_independiente`, `anio_prestamo`, `mes_prestamo`, `dia_semana_prestamo`, `fin_de_mes`, `total_creditos`, `tipo_credito_4`, `tipo_credito_6`, `tipo_credito_7`, `tipo_credito_9`, `tipo_credito_10`, `tendencia_ingresos_Creciente`, `tendencia_ingresos_Decreciente`, `tendencia_ingresos_Estable`, `tendencia_ingresos_Sin_informacion`

---

## Notas de mantenimiento

- Si se reentrenan los modelos, los nuevos `.pkl` deben commitearse al repo para que el deploy en Render los incluya en la imagen Docker.
- Si se agregan o eliminan features, regenerar `feature_names.pkl` y actualizar el schema `InsuranceData` en `model_deploy.py`.
- Render free tier duerme los servicios tras 15 minutos de inactividad. Se recomienda configurar un ping periódico a `/health` con [UptimeRobot](https://uptimerobot.com) para mantener la API activa durante demos.