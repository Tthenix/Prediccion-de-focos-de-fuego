# Predicción de Focos de Fuego Usando Machine Learning

- **Preprocesamiento de Datos:**

  - Escalado y normalización de variables con `StandardScaler`.
  - Creación de etiquetas binarias (fuego/no fuego) según el nivel de confianza.

- **Entrenamiento del Modelo:**

  - Uso de `RandomForestClassifier` con balanceo de clases.
  - División de datos en conjuntos de entrenamiento y prueba.

- **Evaluación del Modelo:**

  - Métricas de clasificación como precisión, recall, y F1-score.
  - Generación de matriz de confusión y reporte gráfico.

- **Visualizaciones:**

  - Gráficos de barras para evaluar el desempeño por clase.
  - Mapas interactivos generados con `Folium` para mostrar los resultados de las predicciones.

- **Predicciones Futuras:**

  - Evaluación de nuevos datos satelitales para identificar zonas de riesgo.

## Tecnologías Utilizadas

- **Librerías de Machine Learning y Procesamiento de Datos:**
  - `pandas`, `numpy`, `scikit-learn`
- **Visualización:**
  - `matplotlib`, `seaborn`, `folium`
- **Análisis y Métricas:**
  - `classification_report`, `confusion_matrix`

## Archivos Incluidos

- `main.py`: Script principal que contiene el flujo del proyecto.
- `classification_report.png`: Gráfico de métricas de clasificación.
- `confusion_matrix.png`: Matriz de confusión visual.
- `Predicciones_Focos_Fuego_Noviembre.csv`: Predicciones generadas para el mes de noviembre.
- `Predicciones_Focos_Fuego_Noviembre_Mapa.html`: Mapa interactivo con las predicciones realizadas.

## Datos de Entrada

Los datos satelitales se encuentran en formato CSV e incluyen las siguientes columnas relevantes:

- `Latitud`, `Longitud`: Coordenadas geográficas.
- `FP_T21`, `FP_T31`: Temperaturas relacionadas con los puntos detectados.
- `FP_Power`: Potencia del foco detectado.
- `SCAN`, `TRACK`: Tamaño de los píxeles detectados.
- `FP_Confidence`: Nivel de confianza del foco detectado.

### Ejemplo de Preprocesamiento:

Se define como "fuego" aquellos registros donde `FP_Confidence ≥ 50`.

## Link del proyecto:

https://prediccion-de-focos-de-fuego-saocom.netlify.app/

5. Los resultados se generarán en los siguientes archivos:

   - `classification_report.png`
   - `confusion_matrix.png`
   - `Predicciones_Focos_Fuego_Noviembre.csv`
   - `Predicciones_Focos_Fuego_Noviembre_Mapa.html`


## Interpretación de Resultados

- **Clase 0 (Sin Fuego):** Alta precisión indica que el modelo clasifica correctamente zonas sin fuego, reduciendo falsos positivos.
- **Clase 1 (Con Fuego):** Buen recall asegura que la mayoría de los focos de fuego son identificados correctamente.

---

Este proyecto fue desarrollado como parte de una iniciativa para abordar problemas ambientales utilizando tecnologías avanzadas.

