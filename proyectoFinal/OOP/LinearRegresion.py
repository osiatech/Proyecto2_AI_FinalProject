
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split as tts

class RegresionLineal:
    def __init__(self, file_path, variable_independiente, variable_dependiente):
        self.file_path = file_path
        self.variable_independiente = variable_independiente
        self.variable_dependiente = variable_dependiente
        self.model = LinearRegression()
    
    def cargar_datos(self):
        datos = pd.read_csv(self.file_path)
        self.X = datos[[self.variable_independiente]].values
        self.y = datos[self.variable_dependiente].values

    def dividir_datos(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = tts(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def entrenar_modelo(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        self.y_pred = self.model.predict(self.X_test)
        self.mse_value = mse(self.y_test, self.y_pred)
        self.mae_value = mae(self.y_test, self.y_pred)
        self.r2_value = r2(self.y_test, self.y_pred)
        print(f"Error Cuadrático Medio (MSE): {self.mse_value:.2f}")
        print(f"Error Absoluto Medio (MAE): {self.mae_value:.2f}")
        print(f"Coeficiente de Determinación (R^2): {self.r2_value:.2f}")

    def visualizar_resultados(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test, self.y_test, color='blue', label='Datos reales')
        plt.plot(self.X_test, self.y_pred, color='red', label='Línea de regresión')
        plt.xlabel(self.variable_independiente)
        plt.ylabel(self.variable_dependiente)
        plt.title(f'Relación entre {self.variable_independiente} y {self.variable_dependiente}')
        plt.legend()
        plt.show()

    def predecir(self, nuevo_dato):
        prediccion = self.model.predict([[nuevo_dato]])
        print(f'Predicción para un usuario con {nuevo_dato} {self.variable_independiente}: {prediccion[0]:.2f}')
        return prediccion[0]

# Uso de la clase
if __name__ == "__main__":
    # Creando una instancia de la clase RegresionLineal e Inicializando los atributos definidos en el constructor.
    regresion = RegresionLineal(
        file_path='dataset_fidelidad.csv', 
        variable_independiente='total_compras', 
        variable_dependiente='total_gastado'
    )
    
    # Ejecutar el flujo de trabajo
    regresion.cargar_datos()
    regresion.dividir_datos()
    regresion.entrenar_modelo()
    regresion.evaluar_modelo()
    regresion.visualizar_resultados()
    
    # Hacer una predicción para un nuevo usuario
    nuevo_usuario = 55  # Total de compras realizadas por el nuevo usuario
    regresion.predecir(nuevo_usuario)
