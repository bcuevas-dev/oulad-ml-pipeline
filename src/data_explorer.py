# Análisis Exploratorio de Datos

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class DataExplorer:
    """Clase para análisis exploratorio de datos (EDA)."""
    def __init__(self):
        self.plt = plt
        self.sns = sns
        self.hypothesis_results = {}

    def single_variable_analysis(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Análisis univariado: describe, histogramas, kurtosis."""
        summary = df.describe(include='all')
        print("\nResumen estadístico:\n", summary)
        if output_dir:
            summary.to_csv(f"{output_dir}/single_variable_summary.csv")
        # Histogramas y kurtosis
        for col in df.select_dtypes(include=[np.number]).columns:
            self.plt.figure()
            df[col].hist(bins=30)
            self.plt.title(f"Histograma de {col}")
            self.plt.xlabel(col)
            self.plt.ylabel("Frecuencia")
            if output_dir:
                self.plt.savefig(f"{output_dir}/hist_{col}.png")
            self.plt.close()
            print(f"Kurtosis de {col}: {df[col].kurtosis():.2f}")

    def dual_variable_analysis(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Análisis bivariado: matriz de correlación y scatter plot de las principales variables."""
        correlation_matrix = df.corr(numeric_only=True)
        print("\nMatriz de correlación:\n", correlation_matrix)
        if output_dir:
            correlation_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")
        # Heatmap de correlación
        self.plt.figure(figsize=(10,8))
        self.sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        self.plt.title("Matriz de correlación")
        if output_dir:
            self.plt.savefig(f"{output_dir}/correlation_heatmap.png")
        self.plt.close()
        # Scatter plot de las dos variables más correlacionadas
        if len(correlation_matrix.columns) >= 2:
            top_correlation = correlation_matrix.abs().unstack().sort_values(ascending=False)
            top_correlation = top_correlation[top_correlation < 1].index[0]
            x_var, y_var = top_correlation
            self.plt.figure()
            self.sns.scatterplot(x=df[x_var], y=df[y_var])
            self.plt.title(f"Scatter plot: {x_var} vs {y_var}")
            if output_dir:
                self.plt.savefig(f"{output_dir}/scatter_{x_var}_vs_{y_var}.png")
            self.plt.close()

    def generate_boxplots(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        """Boxplots para variables numéricas."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.plt.figure()
            self.sns.boxplot(y=df[col])
            self.plt.title(f"Boxplot de {col}")
            if output_dir:
                self.plt.savefig(f"{output_dir}/boxplot_{col}.png")
            self.plt.close()

    # ========================================================================
    # MÉTODOS NUEVOS AGREGADOS PARA EL PROYECTO FINAL - CORREGIDOS
    # ========================================================================

    def test_project_hypotheses(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Prueba las hipótesis específicas del proyecto.
        
        H1: La participación en VLE predice significativamente el resultado académico final
        H2: Estudiantes con mayor edad tienen mejor rendimiento académico
        """
        print("\nTESTEO DE HIPÓTESIS DEL PROYECTO")
        print("-" * 50)
        
        # DEBUG: Mostrar columnas disponibles
        print(f"\nDEBUG - Columnas disponibles:")
        age_columns = [col for col in df.columns if 'age' in col.lower()]
        print(f"  Columnas de edad: {age_columns}")
        print(f"  Tiene 'age_band': {'age_band' in df.columns}")
        print(f"  Tiene 'age_band_ord': {'age_band_ord' in df.columns}")
        
        results = {}
        
        # Crear variable numérica para resultado (compartida para H1 y H2)
        result_encoded = None
        if 'final_result' in df.columns:
            if df['final_result'].dtype == 'object':
                le = LabelEncoder()
                result_encoded = le.fit_transform(df['final_result'])
                print(f"  Codificación final_result: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            else:
                result_encoded = df['final_result']
        
        # H1: Participación VLE vs Resultado académico
        if 'sum_click' in df.columns and 'final_result' in df.columns:
            # Correlación
            correlation = df['sum_click'].corr(pd.Series(result_encoded))
            
            # Test estadístico
            correlation_test = stats.pearsonr(df['sum_click'].dropna(), 
                                            pd.Series(result_encoded)[df['sum_click'].dropna().index])
            
            results['h1'] = {
                'correlation': correlation,
                'p_value': correlation_test[1],
                'significant': correlation_test[1] < 0.05,
                'effect_size': abs(correlation),
                'interpretation': self._interpret_correlation_strength(abs(correlation))
            }
            
            print(f"\nH1 - Participación VLE vs Resultado Académico:")
            print(f"  Correlación: {correlation:.3f}")
            print(f"  P-valor: {correlation_test[1]:.3e}")
            print(f"  Significativo: {'Sí' if results['h1']['significant'] else 'No'}")
            print(f"  Tamaño del efecto: {results['h1']['interpretation']}")
        
        # H2: Edad vs Rendimiento (CORREGIDO - crear age_band_ord si no existe)
        age_variable = None
        
        # Intentar encontrar variable de edad ordinal
        if 'age_band_ord' in df.columns:
            age_variable = 'age_band_ord'
            print(f"  EXITO: Usando age_band_ord existente")
        elif 'age_band' in df.columns:
            # CREAR age_band_ord a partir de age_band
            print(f"  INFO: Creando age_band_ord desde age_band...")
            
            # Mapeo común de age_band a valores ordinales
            age_mappings = [
                # Diferentes formatos posibles
                {'0-35': 1, '35-55': 2, '55<=': 3},
                {'Under 35': 1, '35-55': 2, 'Over 55': 3},
                {'<35': 1, '35-55': 2, '>55': 3},
                {'Young': 1, 'Middle': 2, 'Old': 3},
                {'Low': 1, 'Medium': 2, 'High': 3}
            ]
            
            # Intentar cada mapeo
            df['age_band_ord'] = None
            unique_ages = df['age_band'].unique()
            print(f"    Valores únicos en age_band: {unique_ages}")
            
            for mapping in age_mappings:
                mapped_values = df['age_band'].map(mapping)
                if not mapped_values.isna().all():
                    df['age_band_ord'] = mapped_values
                    print(f"    EXITO: Mapeo exitoso: {mapping}")
                    break
            
            # Si ningún mapeo funciona, usar LabelEncoder
            if df['age_band_ord'].isna().all():
                print(f"    INFO: Usando LabelEncoder como fallback...")
                le_age = LabelEncoder()
                df['age_band_ord'] = le_age.fit_transform(df['age_band'].fillna('Unknown'))
                print(f"    Codificación automática: {dict(zip(le_age.classes_, le_age.transform(le_age.classes_)))}")
            
            age_variable = 'age_band_ord'
        
        # Ejecutar H2 si tenemos variable de edad
        if age_variable and age_variable in df.columns and 'final_result' in df.columns:
            try:
                age_correlation = df[age_variable].corr(pd.Series(result_encoded))
                age_test = stats.pearsonr(df[age_variable].dropna(), 
                                        pd.Series(result_encoded)[df[age_variable].dropna().index])
                
                results['h2'] = {
                    'correlation': age_correlation,
                    'p_value': age_test[1],
                    'significant': age_test[1] < 0.05,
                    'effect_size': abs(age_correlation),
                    'interpretation': self._interpret_correlation_strength(abs(age_correlation))
                }
                
                print(f"\nH2 - Edad vs Rendimiento Académico:")
                print(f"  Variable usada: {age_variable}")
                print(f"  Correlación: {age_correlation:.3f}")
                print(f"  P-valor: {age_test[1]:.3e}")
                print(f"  Significativo: {'Sí' if results['h2']['significant'] else 'No'}")
                print(f"  Tamaño del efecto: {results['h2']['interpretation']}")
                
            except Exception as e:
                print(f"\nERROR en H2: {e}")
                results['h2'] = {'error': str(e)}
        else:
            print(f"\nERROR H2 - No se pudo ejecutar: variable de edad no disponible")
            print(f"    age_variable: {age_variable}")
            print(f"    Columnas de edad disponibles: {age_columns}")
        
        # Guardar resultados
        results_df = pd.DataFrame([
            {
                'hypothesis': 'H1_VLE_Performance',
                'correlation': results.get('h1', {}).get('correlation', np.nan),
                'p_value': results.get('h1', {}).get('p_value', np.nan),
                'significant': results.get('h1', {}).get('significant', False),
                'interpretation': results.get('h1', {}).get('interpretation', 'N/A')
            },
            {
                'hypothesis': 'H2_Age_Performance', 
                'correlation': results.get('h2', {}).get('correlation', np.nan),
                'p_value': results.get('h2', {}).get('p_value', np.nan),
                'significant': results.get('h2', {}).get('significant', False),
                'interpretation': results.get('h2', {}).get('interpretation', 'N/A')
            }
        ])
        
        results_df.to_csv(f"{output_dir}/hypothesis_testing_results.csv", index=False)
        
        # Crear visualizaciones de hipótesis
        self._create_hypothesis_visualizations(df, results, output_dir)
        
        # Resumen final
        print(f"\nRESUMEN DE HIPÓTESIS:")
        if 'h1' in results and 'error' not in results['h1']:
            print(f"  EXITO H1 ejecutada - Correlación: {results['h1']['correlation']:.3f}")
        else:
            print(f"  ERROR H1 falló")
            
        if 'h2' in results and 'error' not in results['h2']:
            print(f"  EXITO H2 ejecutada - Correlación: {results['h2']['correlation']:.3f}")
        else:
            print(f"  ERROR H2 falló")
        
        self.hypothesis_results = results
        return results

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpreta la fuerza de la correlación."""
        if correlation >= 0.7:
            return "Fuerte"
        elif correlation >= 0.5:
            return "Moderada"
        elif correlation >= 0.3:
            return "Débil"
        else:
            return "Muy débil"

    def _create_hypothesis_visualizations(self, df: pd.DataFrame, results: Dict, output_dir: str):
        """Crea visualizaciones específicas para las hipótesis."""
        
        # Gráfico H1: VLE vs Resultado
        if 'sum_click' in df.columns and 'final_result' in df.columns:
            self.plt.figure(figsize=(12, 5))
            
            self.plt.subplot(1, 2, 1)
            for result in df['final_result'].unique():
                data = df[df['final_result'] == result]['sum_click']
                self.plt.hist(data, alpha=0.6, label=result, bins=30)
            self.plt.xlabel('Clicks en VLE')
            self.plt.ylabel('Frecuencia')
            self.plt.title('H1: Distribución de Participación VLE por Resultado')
            self.plt.legend()
            
            self.plt.subplot(1, 2, 2)
            self.sns.boxplot(data=df, x='final_result', y='sum_click')
            self.plt.xticks(rotation=45)
            self.plt.title('H1: Participación VLE por Resultado Final')
            
            self.plt.tight_layout()
            self.plt.savefig(f"{output_dir}/hypothesis_h1_vle_performance.png", dpi=300, bbox_inches='tight')
            self.plt.close()
        
        # Gráfico H2: Edad vs Resultado
        if 'age_band' in df.columns and 'final_result' in df.columns:
            self.plt.figure(figsize=(10, 6))
            
            # Crear tabla de contingencia
            contingency = pd.crosstab(df['age_band'], df['final_result'], normalize='index') * 100
            
            contingency.plot(kind='bar', stacked=False)
            self.plt.title('H2: Distribución de Resultados por Grupo de Edad')
            self.plt.xlabel('Grupo de Edad')
            self.plt.ylabel('Porcentaje')
            self.plt.xticks(rotation=45)
            self.plt.legend(title='Resultado Final')
            
            self.plt.tight_layout()
            self.plt.savefig(f"{output_dir}/hypothesis_h2_age_performance.png", dpi=300, bbox_inches='tight')
            self.plt.close()

    def comprehensive_eda_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis EDA comprehensivo mejorado."""
        print("\nANÁLISIS EDA COMPREHENSIVO")
        print("=" * 50)
        
        # EDA básico existente
        self.single_variable_analysis(df, output_dir)
        self.dual_variable_analysis(df, output_dir)
        self.generate_boxplots(df, output_dir)
        
        # Análisis adicionales mejorados
        self._advanced_correlation_analysis(df, output_dir)
        self._target_variable_analysis(df, output_dir)
        self._feature_importance_analysis(df, output_dir)

    def _advanced_correlation_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis de correlación avanzado."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Crear mapa de calor mejorado
            self.plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            self.sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                           cmap='RdBu_r', center=0, square=True,
                           linewidths=0.5, cbar_kws={"shrink": .8})
            self.plt.title('Matriz de Correlación Avanzada - Variables Numéricas')
            self.plt.tight_layout()
            self.plt.savefig(f"{output_dir}/advanced_correlation_matrix.png", dpi=300, bbox_inches='tight')
            self.plt.close()
            
            # Análisis de correlaciones fuertes
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if strong_correlations:
                strong_corr_df = pd.DataFrame(strong_correlations)
                strong_corr_df.to_csv(f"{output_dir}/strong_correlations.csv", index=False)

    def _target_variable_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis específico de la variable objetivo."""
        if 'final_result' in df.columns:
            self.plt.figure(figsize=(15, 10))
            
            # Distribución de la variable objetivo
            self.plt.subplot(2, 3, 1)
            df['final_result'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            self.plt.title('Distribución de Resultados Finales')
            
            # Análisis por género
            if 'gender' in df.columns:
                self.plt.subplot(2, 3, 2)
                pd.crosstab(df['gender'], df['final_result'], normalize='index').plot(kind='bar')
                self.plt.title('Resultados por Género')
                self.plt.xticks(rotation=0)
            
            # Análisis por edad
            if 'age_band' in df.columns:
                self.plt.subplot(2, 3, 3)
                pd.crosstab(df['age_band'], df['final_result'], normalize='index').plot(kind='bar')
                self.plt.title('Resultados por Edad')
                self.plt.xticks(rotation=45)
            
            # Análisis por región
            if 'region' in df.columns:
                self.plt.subplot(2, 3, 4)
                region_results = pd.crosstab(df['region'], df['final_result'])
                top_regions = region_results.sum(axis=1).nlargest(8).index
                region_results.loc[top_regions].plot(kind='bar')
                self.plt.title('Resultados por Top 8 Regiones')
                self.plt.xticks(rotation=45)
            
            # Análisis por nivel educativo
            if 'highest_education' in df.columns:
                self.plt.subplot(2, 3, 5)
                pd.crosstab(df['highest_education'], df['final_result'], normalize='index').plot(kind='bar')
                self.plt.title('Resultados por Nivel Educativo')
                self.plt.xticks(rotation=45)
            
            # Análisis de participación VLE
            if 'sum_click' in df.columns:
                self.plt.subplot(2, 3, 6)
                df.boxplot(column='sum_click', by='final_result', ax=self.plt.gca())
                self.plt.title('Participación VLE por Resultado')
                self.plt.suptitle('')  # Remover título automático
            
            self.plt.tight_layout()
            self.plt.savefig(f"{output_dir}/target_variable_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
            self.plt.close()

    def _feature_importance_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis preliminar de importancia de características."""
        if 'final_result' in df.columns:
            # Preparar datos
            X = df.select_dtypes(include=[np.number])
            if X.shape[1] == 0:
                return
            
            y = df['final_result']
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Rellenar valores nulos
            X = X.fillna(X.median())
            
            # Selección de características con SelectKBest
            selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Obtener puntuaciones
            scores = selector.scores_
            feature_names = X.columns[selector.get_support()]
            
            # Crear gráfico de importancia
            self.plt.figure(figsize=(10, 8))
            feature_scores = pd.DataFrame({
                'feature': feature_names,
                'score': scores[selector.get_support()]
            }).sort_values('score', ascending=True)
            
            self.plt.barh(range(len(feature_scores)), feature_scores['score'])
            self.plt.yticks(range(len(feature_scores)), feature_scores['feature'])
            self.plt.xlabel('F-Score')
            self.plt.title('Importancia Preliminar de Características (SelectKBest)')
            self.plt.tight_layout()
            self.plt.savefig(f"{output_dir}/preliminary_feature_importance.png", dpi=300, bbox_inches='tight')
            self.plt.close()
            
            # Guardar resultados
            feature_scores.to_csv(f"{output_dir}/preliminary_feature_scores.csv", index=False)

    def create_advanced_scatter_plots(self, df: pd.DataFrame, output_dir: str):
        """Crea scatter plots avanzados para múltiples relaciones."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Identificar las correlaciones más interesantes
            corr_matrix = df[numeric_cols].corr()
            interesting_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.3:  # Correlaciones moderadas o fuertes
                        interesting_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            # Ordenar por correlación más fuerte
            interesting_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Crear gráficos para los top 6 pares
            n_plots = min(6, len(interesting_pairs))
            if n_plots > 0:
                fig, axes = self.plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                for i, (var1, var2, corr_val) in enumerate(interesting_pairs[:n_plots]):
                    axes[i].scatter(df[var1], df[var2], alpha=0.6)
                    axes[i].set_xlabel(var1)
                    axes[i].set_ylabel(var2)
                    axes[i].set_title(f'{var1} vs {var2}\nCorrelación: {corr_val:.3f}')
                    
                    # Línea de tendencia
                    z = np.polyfit(df[var1].dropna(), df[var2].dropna(), 1)
                    p = np.poly1d(z)
                    axes[i].plot(df[var1], p(df[var1]), "r--", alpha=0.8)
                    axes[i].grid(True, alpha=0.3)
                
                # Ocultar plots vacíos
                for i in range(n_plots, len(axes)):
                    axes[i].set_visible(False)
                
                self.plt.tight_layout()
                self.plt.savefig(f"{output_dir}/advanced_scatter_plots.png", dpi=300, bbox_inches='tight')
                self.plt.close()

    def generate_distribution_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis detallado de distribuciones."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Análisis de normalidad
            normality_results = []
            
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) > 3:
                    # Test de Shapiro-Wilk para muestras pequeñas
                    if len(data) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(1000, len(data))))
                    else:
                        shapiro_stat, shapiro_p = np.nan, np.nan
                    
                    # Test de Kolmogorov-Smirnov
                    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    
                    normality_results.append({
                        'variable': col,
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'normal_shapiro': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
                        'normal_ks': ks_p > 0.05,
                        'skewness': data.skew(),
                        'kurtosis': data.kurtosis()
                    })
            
            # Guardar resultados de normalidad
            if normality_results:
                normality_df = pd.DataFrame(normality_results)
                normality_df.to_csv(f"{output_dir}/normality_tests.csv", index=False)
                
                print("\nAnálisis de Normalidad:")
                for result in normality_results:
                    print(f"  {result['variable']}:")
                    if result['normal_shapiro'] is not None:
                        print(f"    Shapiro-Wilk: {'Normal' if result['normal_shapiro'] else 'No normal'} (p={result['shapiro_p_value']:.4f})")
                    print(f"    Kolmogorov-Smirnov: {'Normal' if result['normal_ks'] else 'No normal'} (p={result['ks_p_value']:.4f})")

    def generate_outlier_analysis(self, df: pd.DataFrame, output_dir: str):
        """Análisis detallado de outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            outlier_summary = []
            
            for col in numeric_cols:
                data = df[col].dropna()
                
                # Método IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                # Método Z-score
                z_scores = np.abs(stats.zscore(data))
                z_outliers = data[z_scores > 3]
                
                outlier_summary.append({
                    'variable': col,
                    'total_values': len(data),
                    'iqr_outliers': len(iqr_outliers),
                    'iqr_percentage': (len(iqr_outliers) / len(data)) * 100,
                    'z_outliers': len(z_outliers),
                    'z_percentage': (len(z_outliers) / len(data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
            
            # Guardar resumen de outliers
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df.to_csv(f"{output_dir}/outlier_analysis.csv", index=False)
            
            print("\nAnálisis de Outliers:")
            for result in outlier_summary:
                print(f"  {result['variable']}:")
                print(f"    IQR: {result['iqr_outliers']} outliers ({result['iqr_percentage']:.1f}%)")
                print(f"    Z-score: {result['z_outliers']} outliers ({result['z_percentage']:.1f}%)")