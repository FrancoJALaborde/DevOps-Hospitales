<h1 align="center">Análisis de complicaciones infecciosas después de la biopsia</h1>

<p>Este es un proyecto de análisis de datos que se centra en predecir la aparición de complicaciones infecciosas después de una biopsia. El objetivo es utilizar un modelo de aprendizaje automático basado en Random Forest para realizar la predicción.</p>

<h2>Paso 1: Importar bibliotecas</h2>

<pre><code>import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
</code></pre>

<p>En este paso, se importan las bibliotecas necesarias para el análisis y modelado de datos.</p>

<h2>Paso 2: Cargar los datos</h2>

<pre><code>data = pd.read_csv("M6\Proyecto Integrador\Propuesta 1\BBDD_Hospitalizacion - copia.csv", delimiter=";")
</code></pre>

<p>Se carga el conjunto de datos desde un archivo de Excel y se almacena en un DataFrame de pandas.</p>

<h2>Paso 3: Explorar los datos</h2>

<pre><code>print(data.head())
</code></pre>

<p>Aquí se muestra una vista previa de los datos cargados para obtener una idea de su estructura y contenido.</p>

<h2>Paso 4: Preparar los datos para el modelado</h2>

<pre><code>X = data.drop("NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA", axis=1)
y = data["NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

le = LabelEncoder()
for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
</code></pre>

<p>En este paso, se preparan los datos para el modelado. Se divide el conjunto de datos en conjuntos de entrenamiento y prueba, y se codifican las variables categóricas.</p>

<h2>Paso 5: Entrenar el modelo Random Forest</h2>

<pre><code>rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
</code></pre>

<p>Se entrena un modelo Random Forest utilizando los datos de entrenamiento.</p>

<h2>Paso 6: Evaluar el rendimiento del modelo</h2>

<pre><code>y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
</code></pre>

<p>Se evalúa el rendimiento del modelo utilizando métricas de clasificación.</p>

<h2>Paso 7: Identificar las características más importantes del modelo</h2>

<pre><code>importances = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
plt.bar(importances["Feature"], importances["Importance"])
plt.xticks(rotation=90)
plt.show()
</code></pre>

<p>Se identifican las características más importantes del modelo y se muestra un gráfico de barras para visualizar su importancia.</p>
