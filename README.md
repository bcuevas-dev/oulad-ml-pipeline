# OULAD Machine Learning Pipeline

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para el análisis del dataset **OULAD (Open University Learning Analytics Dataset)**. El sistema realiza análisis predictivo del rendimiento académico utilizando técnicas de aprendizaje automático y análisis exploratorio de datos.

## Objetivo

Desarrollar un sistema de análisis predictivo que permita:
- Predecir el rendimiento académico de los estudiantes
- Identificar factores clave que influyen en el éxito académico
- Generar insights mediante análisis exploratorio de datos
- Proporcionar herramientas de interpretabilidad del modelo

## Estructura del Proyecto

```
ML_Proyect/
├── config/
│   └── settings.py              # Configuración de la base de datos
├── notebooks/
│   └── oulad_pipeline_visual_final.ipynb  # Notebook interactivo completo
├── src/
│   ├── connection_manager.py    # Gestión de conexiones a BD
│   ├── data_processor.py        # Procesamiento de datos
│   ├── data_explorer.py         # Análisis exploratorio
│   ├── query_executor.py        # Ejecución de consultas SQL
│   ├── helpers.py               # Funciones auxiliares
│   ├── oulad_pipeline.py        # Pipeline principal (Ejecusion del Proyecto)
│   ├── models/
│   │   └── model.py            # Modelos de ML
│   └── utilities/
│       └── data_handlers.py    # Manejo de datos
├── results/                    # Outputs del EDA
├── results_final/              # Outputs de modelos
├── project_requirements.txt    # Dependencias
├── esquema_ouladdb.sql        # Schema de la base de datos
└── README.md                  # Este archivo
```

## 🔧 Instalación y Configuración

### 1. Prerrequisitos
- Python 3.8 o superior
- MySQL Server
- Git

### 2. Instalación de Dependencias
```bash
pip install -r project_requirements.txt
```

### 3. Configuración de la Base de Datos
1. Importar el esquema: `mysql -u usuario -p database_name < esquema_ouladdb.sql`
2. Configurar credenciales en `config/settings.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'tu_usuario',
    'password': 'tu_password',
    'database': 'oulad_db'
}
```

### 4. Datos del Dataset OULAD
El dataset debe ser cargado en MySQL con las tablas:
- `studentInfo`: Información demográfica de estudiantes
- `studentVle`: Interacciones con el entorno virtual
- `courses`: Información de cursos
- `assessments`: Información de evaluaciones
- `studentAssessment`: Resultados de evaluaciones

## Uso del Sistema

### Opción 1: Pipeline Completo (Recomendado)
```bash
python src/oulad_pipeline.py
```

### Opción 2: Con Límite de Datos (Para Pruebas)
```bash
python src/oulad_pipeline.py --max_rows 10000 --strategy balanced --quick
```

### Opción 3: Notebook Interactivo
```bash
jupyter notebook notebooks/oulad_pipeline_visual_final.ipynb
```

## Funcionalidades Principales

### Análisis Exploratorio de Datos (EDA)
- **Análisis univariado**: Distribuciones, histogramas, estadísticas descriptivas
- **Análisis bivariado**: Correlaciones, scatter plots, análisis de grupos
- **Visualizaciones**: Heatmaps, boxplots, gráficos de barras
- **Pruebas de hipótesis**: Tests estadísticos para validar relaciones

### Modelos de Machine Learning
- **Random Forest**: Modelo ensemble para clasificación
- **Gradient Boosting**: Algoritmo de boosting avanzado
- **Logistic Regression**: Modelo lineal interpretable
- **Support Vector Machine**: Clasificador con kernel RBF
- **Neural Network**: Red neuronal multicapa

### Métricas y Evaluación
- **Métricas de clasificación**: Accuracy, Precision, Recall, F1-Score
- **Matriz de confusión**: Visualización de errores de clasificación
- **Importancia de características**: Ranking de variables más relevantes
- **Curvas ROC**: Análisis de rendimiento del modelo

### Interpretabilidad
- **Feature Importance**: Importancia de variables por modelo
- **SHAP Values**: Explicaciones locales y globales (opcional)
- **Análisis de correlaciones**: Matrices de correlación
- **Visualizaciones interpretables**: Gráficos explicativos

## Outputs del Sistema

### Carpeta `results/` (EDA)
- `single_variable_summary.csv`: Resumen estadístico univariado
- `correlation_matrix.csv`: Matriz de correlaciones
- `correlation_heatmap.png`: Heatmap de correlaciones
- `hist_*.png`: Histogramas de variables
- `hypothesis_testing_results.csv`: Resultados de pruebas de hipótesis

### Carpeta `results_final/` (Modelos)
- `model_*.joblib`: Modelos entrenados guardados
- `performance_metrics_complete.csv`: Métricas de todos los modelos
- `model_comparison_report.csv`: Comparación de modelos
- `feature_importances.csv`: Importancia de características
- `feature_importance.png`: Gráfico de importancia
- `confusion_matrix.png`: Matriz de confusión
- `test_predictions_complete.csv`: Predicciones del conjunto de prueba

## Arquitectura Técnica

### Clases Principales
- **`DatabaseManager`**: Gestión de conexiones MySQL
- **`DataProcessor`**: Limpieza y transformación de datos
- **`DataExplorer`**: Análisis exploratorio y visualizaciones
- **`OULADModel`**: Entrenamiento y evaluación de modelos
- **`ResultsAnalyzer`**: Análisis de resultados y métricas

### Pipeline de Ejecución
1. **Obtener datos**: Conexión a BD y extracción
2. **Scrub (Limpiar)**: Tratamiento de nulos, outliers, duplicados
3. **Explore (Explorar)**: EDA completo con visualizaciones
4. **Model (Modelar)**: Entrenamiento de múltiples algoritmos
5. **iNterpret (Interpretar)**: Análisis de resultados y explicabilidad

## Dependencias

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
mysql-connector-python>=8.0.0
tqdm>=4.62.0
imbalanced-learn>=0.8.0
sqlalchemy>=1.4.0
xgboost>=1.5.0
scipy>=1.7.0
joblib>=1.1.0
plotly>=5.0.0
statsmodels>=0.12.0
```

## Rendimiento y Optimización

- **Optimización de memoria**: Reducción automática del uso de RAM
- **Procesamiento por lotes**: Manejo eficiente de grandes datasets
- **Paralelización**: Uso de múltiples cores para entrenamiento
- **Caching**: Guardado de resultados intermedios

## Características Avanzadas

### Manejo de Datos Desbalanceados
- **SMOTE**: Sobremuestreo sintético para balancear clases
- **Métricas balanceadas**: F1-Score, Precision, Recall ponderados

### Validación Robusta
- **Validación cruzada**: Evaluación robusta de modelos
- **Grid Search**: Búsqueda de hiperparámetros óptimos
- **Métricas múltiples**: Evaluación desde diferentes perspectivas

### Reproducibilidad
- **Seeds fijas**: Resultados reproducibles
- **Logging completo**: Registro detallado de ejecución
- **Versionado de modelos**: Guardado de modelos entrenados

## Casos de Uso

1. **Análisis Predictivo**: Predecir qué estudiantes están en riesgo de abandono
2. **Investigación Educativa**: Identificar factores que influyen en el rendimiento
3. **Optimización Curricular**: Analizar efectividad de diferentes módulos
4. **Intervención Temprana**: Detectar estudiantes que necesitan apoyo adicional

## Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Ing. Bienvenido Cuevas-**
- **Ing. Ayzel Mateo-**
- **Licda. Ana Esther Segura Reyes-**

## Agradecimientos

- Open University por proporcionar el dataset OULAD
- Comunidad de scikit-learn por las herramientas de ML
- Contribuidores del proyecto

##  Contacto

Para preguntas o soporte:

- Issues: [GitHub Issues](https://github.com/usuario/oulad-ml-pipeline/issues)

---

**¡Gracias por usar el OULAD Machine Learning Pipeline!**
