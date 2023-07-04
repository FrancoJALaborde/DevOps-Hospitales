<h1 align="center">Analysis of Infectious Complications after Biopsy</h1>
<p>This is a data analysis project that focuses on predicting the occurrence of infectious complications after a biopsy. The goal is to use a Random Forest-based machine learning model for prediction.</p>
<h2>Step 1: Import Libraries</h2>
<pre><code>import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
</code></pre>
<p>In this step, the necessary libraries for data analysis and modeling are imported.</p>
<h2>Step 2: Load the Data</h2>
<pre><code>data = pd.read_csv("M6\Proyecto Integrador\Propuesta 1\BBDD_Hospitalizacion - copia.csv", delimiter=";")
</code></pre>
<p>The dataset is loaded from an Excel file and stored in a pandas DataFrame.</p>
<h2>Step 3: Explore the Data</h2>
<pre><code>print(data.head())
</code></pre>
<p>Here, a preview of the loaded data is shown to get an idea of its structure and content.</p>
<h2>Step 4: Prepare the Data for Modeling</h2>
<pre><code>X = data.drop("NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA", axis=1)
y = data["NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACION INFECCIOSA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

le = LabelEncoder()
for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
</code></pre>
<p>In this step, the data is prepared for modeling. The dataset is divided into training and test sets, and categorical variables are encoded.</p>
<h2>Step 5: Train the Random Forest Model</h2>
<pre><code>rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
</code></pre>
<p>A Random Forest model is trained using the training data.</p>
<h2>Step 6: Evaluate the Model Performance</h2>
<pre><code>y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
</code></pre>
<p>The model's performance is evaluated using classification metrics.</p>
<h2>Step 7: Identify the Most Important Features of the Model</h2>
<pre><code>importances = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
importances = importances.sort_values("Importance", ascending=False)
plt.bar(importances["Feature"], importances["Importance"])
plt.xticks(rotation=90)
plt.show()
</code></pre>
<p>The most important features of the model are identified, and a bar graph is shown to visualize their importance.</p>
