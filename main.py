import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Inicializar la app y cargar el modelo UNA SOLA VEZ al arrancar
app = FastAPI(title="Monitor Térmico MLOps")
modelo = joblib.load("modelo_termico.joblib")

# 2. Definir el contrato de entrada (Pydantic)
class DatosHardware(BaseModel):
    temperatura: float
    carga_porcentaje: float

# 3. Endpoint de predicción
@app.post("/predecir")
def predecir_throttling(datos: DatosHardware):
    # El modelo espera una lista de listas: [[temp, carga]]
    entrada = [[datos.temperatura, datos.carga_porcentaje]]

    # Hacemos la predicción (devuelve un array de numpy, tomamos el primer valor)
    prediccion = int(modelo.predict(entrada)[0])
    probabilidad_throttling = float(modelo.predict_proba(entrada)[0][1])

    return {
        "thermal_throttling": bool(prediccion),
        "probabilidad_riesgo": round(probabilidad_throttling * 100, 2),
        "estado": "PELIGRO: Bajando frecuencias" if prediccion == 1 else "Operación normal"
    }