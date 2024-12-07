
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Cargando el dataset para ser utilizado:
datos_dataset = pd.read_csv('dataset_fidelidad.csv')
total_registros = datos_dataset.shape[0]
print("Total de registros en el dataset:", total_registros)


# Contando los registros por categoría
conteo_categorias = datos_dataset["tipo_cliente"].value_counts()

# Imprimir los resultados
for categoria, cantidad in conteo_categorias.items():
    print(f"Categoría {categoria}: {cantidad} registros")


# Seleccionar variables independientes y dependiente
X = datos_dataset[['edad', 'sexo', 'total_gastado', 'total_compras', 'frecuencia_compras','promedio_gasto_por_compra']]
y = datos_dataset['tipo_cliente']


# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Crear y entrenar el modelo
modelo = LogisticRegression(max_iter=100) # iteraciones sobre la data de entrenamiento
modelo.fit(X_train, y_train)


# Realizar predicciones
y_pred = modelo.predict(X_test)
# Evaluar el modelo
print("Precision del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificacion:\n", classification_report(y_test, y_pred, target_names=["Basico", "Oro", "Premium"]))


# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)


# Mostrar los valores de la matriz
print("Matriz de Confusion:\n", matriz_confusion)


nuevo_dato = {
    "edad": 47,
    "sexo": 0,  # Femenino
    "total_gastado": 7329,
    "total_compras": 52,
    "frecuencia_compras": 12,
    "promedio_gasto_por_compra": 155
}

# Convertir en DataFrame
#nuevo_dato_df = pd.DataFrame([nuevo_dato])

# Realizar predicción
#prediccion = modelo.predict(nuevo_dato_df)

# Interpretar la predicción
#mapa_clases = {0: "Basico", 1: "Oro", 2: "Premium"}
#prediccion_interpretada = mapa_clases[prediccion[0]]
#print("Nivel de fidelidad predicho:", prediccion_interpretada)


# Datos de varios usuarios
nuevos_datos = [
    {"edad": 28, "sexo": 1, "total_gastado": 200, "total_compras": 10, "frecuencia_compras": 5, "promedio_gasto_por_compra": 66.67},

    {"edad": 26, "sexo": 0, "total_gastado": 1500, "total_compras": 19, "frecuencia_compras": 5, "promedio_gasto_por_compra": 0},

    {"edad": 25, "sexo": 0, "total_gastado": 3222, "total_compras": 40, "frecuencia_compras": 10,"promedio_gasto_por_compra": 75}
]


# Convertir en DataFrame
nuevos_datos_df = pd.DataFrame(nuevos_datos)

# Realizar predicción para todos los usuarios
predicciones = modelo.predict(nuevos_datos_df)

# Interpretar las predicciones
mapa_clases = {0: "Basico", 1: "Oro", 2: "Premium"}


# Mostrar resultados
for i, pred in enumerate(predicciones):
    prediccion_interpretada = mapa_clases[pred]
    print(f"Usuario {i+1} - Nivel de fidelidad predicho: {prediccion_interpretada}")



# Contar cuántos usuarios hay en cada nivel de fidelidad
nivel_fidelidad = ['Basico', 'Oro', 'Premium']
conteo_niveles = datos_dataset['tipo_cliente'].value_counts()

# Crear una figura con dos subgráficas (una fila, dos columnas)
plt.figure(figsize=(12, 6))  # Tamaño de la figura

# Gráfico de barras (primer gráfico)
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, gráfico 1
plt.bar(nivel_fidelidad, conteo_niveles, color=["cyan", "orange", "lime"])
plt.title('Distribución de los Niveles de Fidelidad en el Dataset (Barras)')
plt.xlabel('Niveles de Fidelidad')
plt.ylabel('Cantidad de Usuarios')

# Gráfico de pastel (segundo gráfico)
plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, gráfico 2
plt.pie(conteo_niveles, labels=nivel_fidelidad, autopct='%1.1f%%', colors=["cyan", "orange", "lime"])
plt.title('Distribución de los Niveles de Fidelidad en el Dataset (Pastel)')

# Mostrar ambas gráficas
plt.tight_layout()  # Ajustar para evitar que las gráficas se sobrepongan
plt.show()





