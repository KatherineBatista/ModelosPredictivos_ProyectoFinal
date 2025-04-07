import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sns.set(style='whitegrid')
verde = '#2ecc71'

# ================================
# PARTE 1: EXPLORACIÓN Y VISUALIZACIÓN (EDA)
# ================================

# Carga del dataset
file = 'C:/Users/RCM67KATBA/Desktop/Katherine Batista Escritorio/Dataset_financial_risk.xlsx'
df = pd.read_excel(file)

# Exploración inicial
print("Primeras filas del dataset:")
print(df.head())

print("\nDimensiones del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Limpieza de datos
df_clean = df.dropna()
print("\nDimensiones después de eliminar nulos:", df_clean.shape)

# Asignación de nivel de riesgo
def clasificar_riesgo(row):
    if row['Previous Defaults'] >= 3 or row['Income'] < 40000:
        return 'Alto'
    elif row['Credit Score'] < 680:
        return 'Medio'
    else:
        return 'Bajo'

df_clean.loc[:, 'Risk Rating'] = df_clean.apply(clasificar_riesgo, axis=1)

print("\nDistribución de niveles de riesgo:")
print(df_clean['Risk Rating'].value_counts())

# Visualizaciones Generales

# Gráfica 1: Distribución de edad
plt.figure(figsize=(8, 4))
sns.histplot(df_clean['Age'], kde=True, bins=20, color=verde)
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('grafica_edad.png')
plt.close()

# Gráfica 2: Ingresos por nivel educativo
plt.figure(figsize=(10, 5))
sns.boxplot(x='Education Level', y='Income', data=df_clean, color=verde)
plt.title('Ingresos por Nivel Educativo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafica_ingresos_educacion.png')
plt.close()

# Gráfica 3: Ingresos por número de dependientes
plt.figure(figsize=(8, 5))
sns.boxplot(x='Number of Dependents', y='Income', data=df_clean, color=verde)
plt.title('Distribución de Ingresos según Número de Dependientes')
plt.tight_layout()
plt.savefig('grafica_ingresos_dependientes_boxplot.png')
plt.close()

# Gráfica 4: Matriz de correlación
plt.figure(figsize=(10, 8))
correlation = df_clean.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='Greens', fmt=".2f")
plt.title('Matriz de Correlación entre Variables Numéricas')
plt.tight_layout()
plt.savefig('matriz_correlacion.png')
plt.close()

# Gráfica 5: Defaults previos por nivel de riesgo
plt.figure(figsize=(8, 5))
sns.boxplot(x='Risk Rating', y='Previous Defaults', data=df_clean, palette=[verde]*3)
plt.title('Defaults Previos por Nivel de Riesgo')
plt.tight_layout()
plt.savefig('grafica_defaults_riesgo.png')
plt.close()

# Gráfica 6: Ingresos por nivel de riesgo
plt.figure(figsize=(8, 5))
sns.boxplot(x='Risk Rating', y='Income', data=df_clean, palette=[verde]*3)
plt.title('Ingresos por Nivel de Riesgo')
plt.tight_layout()
plt.savefig('grafica_ingresos_riesgo.png')
plt.close()

# Gráfica 7: Credit Score por nivel de riesgo
plt.figure(figsize=(8, 5))
sns.boxplot(x='Risk Rating', y='Credit Score', data=df_clean, palette=[verde]*3)
plt.title('Credit Score por Nivel de Riesgo')
plt.tight_layout()
plt.savefig('grafica_score_riesgo.png')
plt.close()

# ================================
# PARTE 2: MODELADO PREDICTIVO
# ================================

# Codificación de la variable objetivo
le = LabelEncoder()
df_clean['Risk_Level'] = le.fit_transform(df_clean['Risk Rating'])  # 'Alto'=0, 'Bajo'=1, 'Medio'=2 

# Selección de variables predictoras
features = ['Income', 'Credit Score', 'Previous Defaults', 'Debt-to-Income Ratio', 'Loan Amount', 'Assets Value']
X = df_clean[features]
y = df_clean['Risk_Level']

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================
# ENTRENAMIENTO DE MODELOS
# ================================

# Modelo 1: Regresión Logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

# Modelo 2: Árbol de Decisión
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Modelo 3: Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ================================
# EVALUACIÓN DE MODELOS
# ================================

# Reportes
print("=== Regresión Logística ===")
print(classification_report(y_test, y_pred_log, target_names=le.classes_))

print("=== Árbol de Decisión ===")
print(classification_report(y_test, y_pred_tree, target_names=le.classes_))

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

print("Matriz de Confusión - Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# Guardar el dataset limpio con la variable 'Risk Rating' incluida
df_clean.to_csv('dataset_financial_risk_limpio.csv', index=False)
print("✅ Dataset limpio exportado como 'dataset_financial_risk_limpio.csv'")
