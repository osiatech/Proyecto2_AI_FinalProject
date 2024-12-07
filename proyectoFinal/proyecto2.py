import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split as tts

# Definición de la clase para regresión lineal
class RegresionLineal:
    def __init__(self, file_path, variable_independiente, variable_dependiente):
        # Inicializa los atributos con los parámetros proporcionados
        self.file_path = file_path
        self.variable_independiente = variable_independiente
        self.variable_dependiente = variable_dependiente
        self.model = LinearRegression()  # Crea un modelo de regresión lineal
    
    def cargar_datos(self):
        # Carga los datos desde el archivo CSV
        datos = pd.read_csv(self.file_path)
        self.X = datos[[self.variable_independiente]].values  # Variable independiente
        self.y = datos[self.variable_dependiente].values  # Variable dependiente

    def dividir_datos(self, test_size=0.3, random_state=42):
        # Divide los datos en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = tts(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def entrenar_modelo(self):
        # Entrena el modelo con los datos de entrenamiento
        self.model.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        # Realiza predicciones con el conjunto de prueba y evalúa el modelo
        self.y_pred = self.model.predict(self.X_test)
        self.mse_value = mse(self.y_test, self.y_pred)  # Error cuadrático medio
        self.mae_value = mae(self.y_test, self.y_pred)  # Error absoluto medio
        self.r2_value = r2(self.y_test, self.y_pred)  # Coeficiente de determinación (R^2)
        # Imprime las métricas de evaluación
        print(f"Error Cuadrático Medio (MSE): {self.mse_value:.2f}")
        print(f"Error Absoluto Medio (MAE): {self.mae_value:.2f}")
        print(f"Coeficiente de Determinación (R^2): {self.r2_value:.2f}")

    def visualizar_resultados(self):
        # Visualiza los resultados con un gráfico de dispersión y la línea de regresión
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test, self.y_test, color='blue', label='Datos reales')
        plt.plot(self.X_test, self.y_pred, color='red', label='Línea de regresión')
        plt.xlabel(self.variable_independiente)
        plt.ylabel(self.variable_dependiente)
        plt.title(f'Relación entre {self.variable_independiente} y {self.variable_dependiente}')
        plt.legend()
        plt.show()

    def predecir(self, nuevo_dato):
        # Realiza una predicción para un nuevo dato
        prediccion = self.model.predict([[nuevo_dato]])
        print(f'Predicción para un usuario con {nuevo_dato} {self.variable_independiente}: {prediccion[0]:.2f}')
        return prediccion[0]

# Uso de la clase
if __name__ == "__main__":
    # Crea una instancia de la clase RegresionLineal e inicializa los atributos
    regresion = RegresionLineal(
        file_path='dataset_fidelidad.csv', 
        variable_independiente='total_compras', 
        variable_dependiente='total_gastado'
    )
    
    # Ejecuta el flujo de trabajo
    regresion.cargar_datos()  # Carga los datos
    regresion.dividir_datos()  # Divide los datos en entrenamiento y prueba
    regresion.entrenar_modelo()  # Entrena el modelo
    regresion.evaluar_modelo()  # Evalúa el modelo
    regresion.visualizar_resultados()  # Muestra los resultados visualmente
    
    # Realiza una predicción para un nuevo usuario
    nuevo_usuario = 55  # Total de compras realizadas por el nuevo usuario
    regresion.predecir(nuevo_usuario)

# Modelo de clasificación de fidelidad
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Definición de la clase para modelo de fidelidad
class ModeloFidelidad:
    def __init__(self, file_path):
        # Inicializa los atributos del modelo
        self.file_path = file_path
        self.modelo = LogisticRegression(max_iter=100)
        self.mapa_clases = {0: "Basico", 1: "Oro", 2: "Premium"}
    
    def cargar_datos(self):
        # Carga los datos y muestra estadísticas iniciales
        self.datos = pd.read_csv(self.file_path)
        self.total_registros = self.datos.shape[0]
        print("Total de registros en el dataset:", self.total_registros)
        print("\nDistribución de categorías:")
        print(self.datos["tipo_cliente"].value_counts())
    
    def preparar_datos(self):
        # Define las variables independientes y dependientes y divide los datos
        self.X = self.datos[['edad', 'sexo', 'total_gastado', 'total_compras', 'frecuencia_compras', 'promedio_gasto_por_compra']]
        self.y = self.datos['tipo_cliente']
        
        # Divide en datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
    
    def entrenar_modelo(self):
        # Entrena el modelo de regresión logística
        self.modelo.fit(self.X_train, self.y_train)
    
    def evaluar_modelo(self):
        # Realiza predicciones y evalúa el modelo
        self.y_pred = self.modelo.predict(self.X_test)
        print("Precisión del modelo:", accuracy_score(self.y_test, self.y_pred))
        print("\nReporte de Clasificación:\n", classification_report(self.y_test, self.y_pred, target_names=self.mapa_clases.values()))
        
        # Muestra la matriz de confusión
        matriz_confusion = confusion_matrix(self.y_test, self.y_pred)
        print("Matriz de Confusión:\n", matriz_confusion)
    
    def predecir(self, nuevos_datos):
        # Predice los niveles de fidelidad para nuevos usuarios
        nuevos_datos_df = pd.DataFrame(nuevos_datos)
        predicciones = self.modelo.predict(nuevos_datos_df)
        
        # Muestra los resultados
        resultados = []
        for i, pred in enumerate(predicciones):
            prediccion_interpretada = self.mapa_clases[pred]
            resultados.append((i + 1, prediccion_interpretada))
            print(f"Usuario {i+1} - Nivel de fidelidad predicho: {prediccion_interpretada}")
        return resultados
    
    def graficar_distribucion(self):
        # Grafica la distribución de los niveles de fidelidad
        conteo_niveles = self.datos['tipo_cliente'].value_counts()
        
        # Crea una figura con dos subgráficas
        plt.figure(figsize=(12, 6))
        
        # Gráfico de barras
        plt.subplot(1, 2, 1)
        plt.bar(self.mapa_clases.values(), conteo_niveles, color=["cyan", "orange", "lime"])
        plt.title('Distribución de los Niveles de Fidelidad en el Dataset (Barras)')
        plt.xlabel('Niveles de Fidelidad')
        plt.ylabel('Cantidad de Usuarios')
        
        # Gráfico de pastel
        plt.subplot(1, 2, 2)
        plt.pie(conteo_niveles, labels=self.mapa_clases.values(), autopct='%1.1f%%', colors=["cyan", "orange", "lime"])
        plt.title('Distribución de los Niveles de Fidelidad en el Dataset (Pastel)')
        
        # Muestra las gráficas
        plt.tight_layout()
        plt.show()

# Uso del modelo de fidelidad
if __name__ == "__main__":
    # Crear instancia del modelo de fidelidad
    modelo_fidelidad = ModeloFidelidad(file_path='dataset_fidelidad.csv')
    
    # Ejecuta el flujo de trabajo
    modelo_fidelidad.cargar_datos()  # Carga los datos
    modelo_fidelidad.preparar_datos()  # Prepara los datos para el modelo
    modelo_fidelidad.entrenar_modelo()  # Entrena el modelo
    modelo_fidelidad.evaluar_modelo()  # Evalúa el modelo
    
    # Predice el nivel de fidelidad de nuevos usuarios
    nuevos_usuarios = [
        {"edad": 28, "sexo": 1, "total_gastado": 200, "total_compras": 10, "frecuencia_compras": 5, "promedio_gasto_por_compra": 66.67},
        {"edad": 26, "sexo": 0, "total_gastado": 180, "total_compras": 6, "frecuencia_compras": 2, "promedio_gasto_por_compra": 30.00}
    ]
    modelo_fidelidad.predecir(nuevos_usuarios)  # Predicción para nuevos usuarios
    
    # Gráfica la distribución de los niveles de fidelidad
    modelo_fidelidad.graficar_distribucion()
