import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

class OULADModel:
    """Clase para entrenamiento y evaluación de modelos de ML."""
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.neural_network import MLPClassifier
        self.classifiers = {
            'LogisticRegression': LogisticRegression(n_jobs=-1),
            'SVM': SVC(),
            'KNeighbors': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'DecisionTree': None,  # Puede agregarse si se importa
            'AdaBoost': AdaBoostClassifier(),
            'Bagging': BaggingClassifier(n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_jobs=-1),
            'RandomForest': RandomForestClassifier(n_jobs=-1),
            'MLP': MLPClassifier(max_iter=1024, random_state=1, hidden_layer_sizes=(50,), learning_rate='adaptive'),
        }
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        self.regressors = {
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
        }
        from sklearn.cluster import KMeans
        self.KMeans = KMeans
        self.plt = plt
        self.sns = sns

    # ... Métodos de OULADModel copiados desde oulad_pipeline.py ...
    # Por brevedad, aquí solo se muestra la estructura. Copia todos los métodos de OULADModel aquí.

    def train_regressors(self, X, y, output_dir: str = "results"):
        """Entrena y evalúa regresores estándar sobre los datos proporcionados."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        # Solo usar variables numéricas
        X_reg = X.select_dtypes(include=[np.number])
        if X_reg.shape[1] == 0:
            X_reg = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(n_jobs=-1, random_state=42)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Guardar métricas y predicciones
        pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv(f"{output_dir}/regression_preds.csv", index=False)
        pd.DataFrame([{'mse': mse, 'r2': r2}]).to_csv(f"{output_dir}/regression_metrics.csv", index=False)
        joblib.dump(reg, f"{output_dir}/regressor_model.joblib")
        print(f"Regresión completada. MSE: {mse:.4f}, R2: {r2:.4f}. Resultados exportados a '{output_dir}/'.")

    def train_clustering(self, X, output_dir: str = "results", n_clusters: int = 3):
        """Entrena un modelo de clustering KMeans y exporta etiquetas y métricas básicas."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        # Solo usar variables numéricas
        X_clust = X.select_dtypes(include=[np.number])
        if X_clust.shape[1] == 0:
            X_clust = pd.get_dummies(X, drop_first=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_clust)
        sil_score = silhouette_score(X_clust, labels)
        pd.DataFrame({'cluster': labels}).to_csv(f"{output_dir}/clustering_labels.csv", index=False)
        pd.DataFrame([{'silhouette_score': sil_score}]).to_csv(f"{output_dir}/clustering_metrics.csv", index=False)
        print(f"Clustering completado. Silhouette score: {sil_score:.4f}. Resultados exportados a '{output_dir}/'.")
