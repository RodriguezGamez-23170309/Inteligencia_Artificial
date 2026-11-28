


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. CREAR EL DATASET (12,000 registros)

rng = np.random.default_rng(42)
n = 12000

edad = rng.integers(18, 71, n)
ingreso_mensual = rng.normal(15000, 6000, n).clip(4000, 60000).round(2)
puntaje_crediticio = rng.normal(650, 90, n).clip(300, 850).round(0)
deuda_mensual = (ingreso_mensual * rng.uniform(0.0, 0.7, n)).round(2)
anios_empleo = np.minimum(rng.integers(0, 41, n), edad - 18)
tipo_empleo = rng.choice(['empleado', 'independiente', 'desempleado'],
                         n, p=[0.65, 0.28, 0.07])
tiene_casa = rng.choice([True, False], n, p=[0.45, 0.55])

# Regla "oculta" para decidir si se aprueba el prestamo (mas ruido aleatorio)
ratio_deuda = deuda_mensual / ingreso_mensual
score = (
    (puntaje_crediticio - 650) / 90 * 1.5
    + (ingreso_mensual - 15000) / 6000 * 0.8
    - ratio_deuda * 4
    + anios_empleo * 0.03
    + np.where(tiene_casa, 0.4, 0)
    + np.where(tipo_empleo == 'desempleado', -1.5, 0)
    + np.where(tipo_empleo == 'independiente', -0.2, 0)
    + rng.normal(0, 0.8, n)  # ruido
)
aprobado = np.where(score > 0, 'Si', 'No')

original_df = pd.DataFrame({
    'Edad': edad,
    'Ingreso': ingreso_mensual,
    'Puntaje': puntaje_crediticio,
    'Deuda': deuda_mensual,
    'AniosEmpleo': anios_empleo,
    'TipoEmpleo': tipo_empleo,
    'TieneCasa': tiene_casa,
    'Aprobado': aprobado,
})

print("Registros:", len(original_df))
print(original_df.head(), "\n")
print("Distribucion de la clase:\n", original_df['Aprobado'].value_counts(), "\n")

# Guardar el dataset por si lo necesitas entregar
original_df.to_csv('dataset_prestamos.csv', index=False)


# 2. PREPROCESAMIENTO

df = pd.get_dummies(original_df, columns=['TipoEmpleo'], dtype=int)
df['TieneCasa'] = df['TieneCasa'].astype(int)
df['Aprobado'] = (df['Aprobado'] == 'Si').astype(int)

X, y = df.drop(columns='Aprobado'), df['Aprobado']

# 80% entrenamiento / 20% prueba (estratificado para mantener proporcion de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

# Escalar solo columnas numericas continuas (ajustando solo con train)
cols_num = ['Edad', 'Ingreso', 'Puntaje', 'Deuda', 'AniosEmpleo']
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
X_test[cols_num] = scaler.transform(X_test[cols_num])


# 3. ENTRENAR KNN

k = 5
knn_clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn_clf.fit(X_train, y_train)


# 4. PREDICCION Y EVALUACION

y_pred = knn_clf.predict(X_test)
print(f"Accuracy (k={k}): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Matriz de confusion:\n", confusion_matrix(y_test, y_pred), "\n")
print(classification_report(y_test, y_pred, target_names=['No', 'Si']))


# 5. PROBAR DIFERENTES VALORES DE K

k_list = [1, 3, 5, 7, 9, 11, 15, 21, 31]
resultados = []
for kk in k_list:
    modelo = KNeighborsClassifier(n_neighbors=kk, metric='euclidean')
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test)) * 100
    resultados.append({'k': kk, 'Accuracy (%)': round(acc, 2)})

df_k = pd.DataFrame(resultados)
print("\nComparacion de K:\n", df_k.to_string(index=False))

mejor_k = int(df_k.loc[df_k['Accuracy (%)'].idxmax(), 'k'])
print(f"\nMejor K: {mejor_k}")


# 6. COMPARAR PESOS (uniform vs distance) CON EL MEJOR K

for peso in ['uniform', 'distance']:
    modelo = KNeighborsClassifier(n_neighbors=mejor_k, weights=peso)
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test)) * 100
    print(f"weights='{peso}': {acc:.2f}%")
