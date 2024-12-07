
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


class ModeloFidelidad:
    def __init__(self, file_path):
        self.file_path = file_path
        self.modelo = LogisticRegression(max_iter=100)
        self.mapa_clases = {0: "Basico", 1: "Oro", 2: "Premium"}
    
    def cargar_datos(self):
        # Cargar el dataset
        self.datos = pd.read_csv(self.file_path)
        self.total_registros = self.datos.shape[0]
        print("Total de registros en el dataset:", self.total_registros)
        print("\nDistribución de categorías:")
        print(self.datos["tipo_cliente"].value_counts())
    
    def preparar_datos(self):
        # Definir variables independientes y dependiente
        self.X = self.datos[['edad', 'sexo', 'total_gastado', 'total_compras', 'frecuencia_compras', 'promedio_gasto_por_compra']]
        self.y = self.datos['tipo_cliente']
        
        # Dividir en datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
    
    def entrenar_modelo(self):
        # Entrenar el modelo
        self.modelo.fit(self.X_train, self.y_train)
    
    def evaluar_modelo(self):
        # Realizar predicciones
        self.y_pred = self.modelo.predict(self.X_test)
        
        # Evaluar el modelo
        print("Precisión del modelo:", accuracy_score(self.y_test, self.y_pred))
        print("\nReporte de Clasificación:\n", classification_report(self.y_test, self.y_pred, target_names=self.mapa_clases.values()))
        
        # Mostrar matriz de confusión
        matriz_confusion = confusion_matrix(self.y_test, self.y_pred)
        print("Matriz de Confusión:\n", matriz_confusion)
    
    def predecir(self, nuevos_datos):
        # Convertir datos en DataFrame
        nuevos_datos_df = pd.DataFrame(nuevos_datos)
        
        # Realizar predicciones
        predicciones = self.modelo.predict(nuevos_datos_df)
        
        # Interpretar las predicciones
        resultados = []
        for i, pred in enumerate(predicciones):
            prediccion_interpretada = self.mapa_clases[pred]
            resultados.append((i + 1, prediccion_interpretada))
            print(f"Usuario {i+1} - Nivel de fidelidad predicho: {prediccion_interpretada}")
        return resultados
    
    def graficar_distribucion(self):
        # Contar niveles de fidelidad
        conteo_niveles = self.datos['tipo_cliente'].value_counts()
        
        # Crear una figura con dos subgráficas
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
        
        # Mostrar gráficas
        plt.tight_layout()
        plt.show()


# Uso de la clase
if __name__ == "__main__":
    # Crear instancia del modelo
    modelo_fidelidad = ModeloFidelidad(file_path='dataset_fidelidad.csv')
    
    # Ejecutar flujo de trabajo
    modelo_fidelidad.cargar_datos()
    modelo_fidelidad.preparar_datos()
    modelo_fidelidad.entrenar_modelo()
    modelo_fidelidad.evaluar_modelo()
    
    # Realizar predicciones para nuevos usuarios
    nuevos_usuarios = [
        {"edad": 28, "sexo": 1, "total_gastado": 200, "total_compras": 10, "frecuencia_compras": 5, "promedio_gasto_por_compra": 66.67},
        {"edad": 26, "sexo": 0, "total_gastado": 1500, "total_compras": 19, "frecuencia_compras": 5, "promedio_gasto_por_compra": 0},
        {"edad": 25, "sexo": 0, "total_gastado": 3222, "total_compras": 40, "frecuencia_compras": 10, "promedio_gasto_por_compra": 75}
    ]
    modelo_fidelidad.predecir(nuevos_usuarios)
    
    # Graficar la distribución de niveles de fidelidad
    modelo_fidelidad.graficar_distribucion()
