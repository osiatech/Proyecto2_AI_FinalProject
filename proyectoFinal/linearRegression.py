
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split as tts

# Paso 1: Cargar los datos desde el archivo CSV
file_path = 'dataset_fidelidad.csv'  # Ajusta la ruta a tu archivo
datos_dataset = pd.read_csv(file_path)

# Paso 2: Definir las variables independientes (X) y dependientes (y)
# Usaremos el 'total_compras' como variable independiente y 'total_gastado' como dependiente
X = datos_dataset[['total_compras']].values
y = datos_dataset['total_gastado'].values

# Paso 3: Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

# Paso 4: Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 5: Evaluar el modelo
y_pred = model.predict(X_test)

# Calcular métricas de evaluación
mse_value = mse(y_test, y_pred)
mae_value = mae(y_test, y_pred)
r2_value = r2(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse_value:.2f}")
print(f"Error Absoluto Medio (MAE): {mae_value:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2_value:.2f}")

# Paso 6: Visualizar los resultados de la regresión
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Línea de regresión')
plt.xlabel('Total de Compras')
plt.ylabel('Total Gastado')
plt.title('Relación entre Total de Compras y Total Gastado')
plt.legend()
plt.show()

# Paso 7: Hacer una predicción para un nuevo usuario
nuevo_usuario = [[55]]  # Ejemplo: Total de compras realizadas por un nuevo usuario
gasto_predicho = model.predict(nuevo_usuario)
print(f'Predicción para un usuario con {nuevo_usuario[0][0]} compras: {gasto_predicho[0]:.2f}')
