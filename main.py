import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Funciones auxiliares
def analyze_results(report_dict, y_test, y_pred):
    precision_0 = report_dict['0']['precision'] * 100
    recall_0 = report_dict['0']['recall'] * 100
    f1_score_0 = report_dict['0']['f1-score'] * 100
    precision_1 = report_dict['1']['precision'] * 100
    recall_1 = report_dict['1']['recall'] * 100
    f1_score_1 = report_dict['1']['f1-score'] * 100

    print("\nConclusión para Clase 0 (Sin Fuego):")
    if precision_0 > 90:
        print(f"Alta precisión ({precision_0:.2f}%) indica que el modelo rara vez clasifica erróneamente áreas sin fuego como si tuvieran fuego.")
    else:
        print(f"Moderada precisión ({precision_0:.2f}%) sugiere que el modelo clasifica correctamente muchas áreas sin fuego, pero aún podría mejorar.")
    if recall_0 > 80:
        print(f"El recall alto ({recall_0:.2f}%) indica que el modelo es efectivo en identificar zonas sin fuego, aunque podría aumentar ligeramente.")
    else:
        print(f"Bajo recall ({recall_0:.2f}%) indica que el modelo pierde algunos casos de áreas sin fuego.")
    print(f"F1-Score de la clase 0: {f1_score_0:.2f}%\n")

    print("Conclusión para Clase 1 (Con Fuego):")
    if precision_1 > 90:
        print(f"Alta precisión ({precision_1:.2f}%) sugiere que el modelo clasifica correctamente casi todas las áreas con focos de fuego.")
    else:
        print(f"Moderada precisión ({precision_1:.2f}%) sugiere que el modelo marca algunas áreas incorrectamente como focos de fuego.")
    if recall_1 > 90:
        print(f"El recall alto ({recall_1:.2f}%) indica que el modelo identifica correctamente casi todos los focos de fuego.")
    else:
        print(f"Moderado recall ({recall_1:.2f}%) sugiere que algunos focos de fuego pueden no ser detectados.")
    print(f"F1-Score de la clase 1: {f1_score_1:.2f}%\n")

    cm = confusion_matrix(y_test, y_pred)
    true_negative, false_positive, false_negative, true_positive = cm.ravel()
    print("Interpretación de la Matriz de Confusión:")
    print(f"Verdaderos Negativos (Clase 0 correctamente clasificada como No Fuego): {true_negative}")
    print(f"Falsos Positivos (Clase 0 incorrectamente clasificada como Fuego): {false_positive}")
    print(f"Falsos Negativos (Clase 1 incorrectamente clasificada como No Fuego): {false_negative}")
    print(f"Verdaderos Positivos (Clase 1 correctamente clasificada como Fuego): {true_positive}")

    if false_positive > false_negative:
        print("\nEl modelo tiende a clasificar erróneamente algunas áreas sin fuego como focos de fuego, generando falsos positivos.")
    elif false_negative > false_positive:
        print("\nEl modelo tiende a omitir algunos focos de fuego, generando falsos negativos.")
    else:
        print("\nEl modelo mantiene un buen equilibrio entre falsos positivos y falsos negativos.")

def plot_classification_report(report):
    report_data = []
    for label, metrics in report.items():
        if label in ['0', '1']:
            report_data.append({
                "Clase": label,
                "Precision": metrics['precision'] * 100,
                "Recall": metrics['recall'] * 100,
                "F1-Score": metrics['f1-score'] * 100
            })
    df_report = pd.DataFrame(report_data)
    df_report.set_index('Clase', inplace=True)
    df_report.plot(kind='bar', figsize=(10, 6))
    plt.title("Métricas de Clasificación para cada Clase")
    plt.xlabel("Clase")
    plt.ylabel("Valor (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.savefig('classification_report.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=['No Fuego', 'Fuego'],
                yticklabels=['No Fuego', 'Fuego'])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig('confusion_matrix.png')
    plt.close()

# Cargar y preparar los datos
file_paths = [
    'CONAE__MODIS_FC_20241001_20241023_Conf80_Detalle.csv',
    'CONAE__MODIS_FC_20241010_20241031_Conf20_Detalle.csv',
    'CONAE__MODIS_FC_20241010_20241031_Conf0_Detalle.csv'
]

focos_df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
focos_df['EsFuego'] = np.where(focos_df['FP_Confidence'] >= 50, 1, 0)

features = focos_df[['Latitud', 'Longitud', 'FP_T21', 'FP_T31', 'FP_Power', 'SCAN', 'TRACK']]
target = focos_df['EsFuego']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
report = classification_report(y_test, y_pred)

print(f"Precisión del modelo: {accuracy:.2f}%")
print("\nReporte de clasificación:")
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
analyze_results(report_dict, y_test, y_pred)
plot_classification_report(report_dict)
plot_confusion_matrix(y_test, y_pred)

# Predicciones para noviembre
file_path_november = 'CONAE__MODIS_FC_20241010_20241031_Conf20_Detalle.csv'
november_df = pd.read_csv(file_path_november)
november_features = november_df[['Latitud', 'Longitud', 'FP_T21', 'FP_T31', 'FP_Power', 'SCAN', 'TRACK']]
november_features_scaled = scaler.transform(november_features)

november_predictions = model.predict(november_features_scaled)
november_probabilities = model.predict_proba(november_features_scaled)

november_df['EsFuego_Pred'] = november_predictions
november_df['Probabilidad_No_Fuego'] = november_probabilities[:, 0] * 100
november_df['Probabilidad_Fuego'] = november_probabilities[:, 1] * 100

november_df.to_csv('Predicciones_Focos_Fuego_Noviembre.csv', index=False)
print("Predicciones para noviembre guardadas en 'Predicciones_Focos_Fuego_Noviembre.csv'")

# Crear el mapa
map_center = [november_df['Latitud'].mean(), november_df['Longitud'].mean()]
fire_map = folium.Map(location=map_center, zoom_start=6)

for _, row in november_df.iterrows():
    if row['EsFuego_Pred'] == 1:
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            popup=folium.Popup(f"Probabilidad de Fuego: {row['Probabilidad_Fuego']:.2f}%", parse_html=True)
        ).add_to(fire_map)
    else:
        folium.CircleMarker(
            location=[row['Latitud'], row['Longitud']],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            popup=folium.Popup(f"Probabilidad de No Fuego: {row['Probabilidad_No_Fuego']:.2f}%", parse_html=True)
        ).add_to(fire_map)

fire_map.save('Predicciones_Focos_Fuego_Noviembre_Mapa.html')
print("Mapa de predicciones guardado en 'Predicciones_Focos_Fuego_Noviembre_Mapa.html'")