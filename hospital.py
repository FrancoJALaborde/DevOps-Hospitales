# Paso 1: Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Paso 2: Cargar los datos desde un archivo de Excel en un DataFrame de pandas
data = pd.read_csv("M6\Proyecto Integrador\Propuesta 1\BBDD_Hospitalizacion - copia.csv",delimiter=";")

# Paso 3: Explorar los datos
print(data.head())

# Paso 4: Preparar los datos para el modelado
X = data.drop("NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA", axis=1)
y = data["NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Codificar variables categóricas
le = LabelEncoder()
for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Paso 5: Entrenar el modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Paso 6: Evaluar el rendimiento del modelo
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Paso 7: Identificar las características más importantes del modelo
importances = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
plt.bar(importances["Feature"], importances["Importance"])
plt.xticks(rotation=90)
plt.show()



X = data.drop("NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA", axis=1)
y = data["NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA"]

le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = le.fit_transform(X[col].astype(str))

importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
print(importances)