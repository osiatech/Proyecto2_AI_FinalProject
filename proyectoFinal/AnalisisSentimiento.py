
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')




#Cargamos nuestro dataset
df = pd.read_csv("programa_fidelidad_dataset.csv")

print(df.head())


#---------------------ANALISIS DE SENTIMIENTO-------------------
#-------------------------------------------------------------

#Empezamos limpiando el texto de la reseña
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\W', ' ', texto)  # Eliminar caracteres especiales
    texto = re.sub(r'\d+', '', texto)  # Eliminar números
    palabras = texto.split()
    palabras_limpias = [palabra for palabra in palabras if palabra not in stopwords.words('spanish')]
    return ' '.join(palabras_limpias)
  

df['limpio'] = df['resenas'].apply(limpiar_texto)
print(df[['resenas', 'limpio']])

#Convertimos el texto en una matriz de características usando TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['limpio'])
y = df['sentimiento_resena']

# Dividimos los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamos el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Aqui evaluamos el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])

accuracy, report
print(f"Accuracy del modelo: {accuracy:.2f}")
print("\nReporte de Clasificación:")
print(report)

# --------------------- PROBAR UNA NUEVA RESEÑA ---------------------
nueva_resena = "me gusta el programa"

# Limpiar y vectorizar la nueva reseña
nueva_resena_limpia = limpiar_texto(nueva_resena)
nueva_resena_vectorizada = vectorizer.transform([nueva_resena_limpia])

# Hacer la predicción
prediccion = model.predict(nueva_resena_vectorizada)

# Etiquetas para las clases
etiquetas = {0: "neutral", 1: "negativo", 2: "positivo"}

# Mostrar el resultado
print(f"Reseña: {nueva_resena}")
print(f"Predicción: {etiquetas[prediccion[0]]}")