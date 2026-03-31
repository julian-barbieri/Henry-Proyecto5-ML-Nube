# 1) librerias
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import pandas as pd
import joblib

# 2) instanciamos la aplicacion
app = FastAPI()

# 3) Aca llamamos al modelo ya entrenado
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "RandomForestClassifier_optuna.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
    
# 4) Creamos un modelo de datos de entrada, 
# con el objetivo de validar
class InsuranceData(BaseModel):

    # Numéricas principales
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: int
    total_otros_prestamos: int
    cuota_pactada: int
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int

    # Créditos por sector
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int

    promedio_ingresos_datacredito: float

    # Features derivadas
    grupoEdad: int
    es_independiente: int

    anio_prestamo: int
    mes_prestamo: int
    dia_semana_prestamo: int
    fin_de_mes: int

    total_creditos: int

    # One-hot tipo_credito
    tipo_credito_4: int
    tipo_credito_9: int
    tipo_credito_10: int
    tipo_credito_6: int
    tipo_credito_7: int

    # One-hot tendencia_ingresos
    tendencia_ingresos_Estable: int
    tendencia_ingresos_Creciente: int
    tendencia_ingresos_Decreciente: int
    tendencia_ingresos_Sin_informacion: int
    
# endpoint /health
@app.get("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

# 5) vamos a crearnos el 1er endpoint: /saludo
@app.post("/predict")
def predict(data: InsuranceData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(input_df)[0]
    return {"Predicted_default": int(prediction)}


@app.post("/predict_batch")
def predict_batch(data: List[InsuranceData]):
    input_df = pd.DataFrame([item.dict() for item in data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    predictions = model.predict(input_df)
    return {"Predicted_default": [int(p) for p in predictions]}




