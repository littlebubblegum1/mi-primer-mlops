import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Datos de entrenamiento: [Temperatura_C, Carga_Porcentaje]
X = [
    [45, 20], [50, 35], [60, 50], [65, 70], [72, 80], # Temperaturas normales -> 0
    [88, 90], [92, 95], [95, 99], [98, 100], [85, 85] # Sobrecalentamiento -> 1
]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 2. Crear y entrenar el modelo
modelo = RandomForestClassifier(n_estimators=10, random_state=42)
modelo.fit(X, y)

# 3. Exportar el modelo entrenado a disco
joblib.dump(modelo, "modelo_termico.joblib")
print("Modelo entrenado y guardado con éxito.")