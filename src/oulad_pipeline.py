"""
oulad_pipeline.py

Pipeline para el flujo OSEMN sobre el dataset OULAD.
Compatible con la estructura existente del proyecto.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import gc
from typing import Tuple, Optional, Dict, Any
from sqlalchemy import create_engine
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_score, 
                           recall_score, mean_squared_error, r2_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings

# Imports locales
from connection_manager import DatabaseManager
from query_executor import fetch_table_data, execute_custom_query, get_table_list
from data_processor import DataProcessor
from data_explorer import DataExplorer
from models.model import OULADModel
from helpers import color_azul_destacado, color_rojo_alerta, color_reset, display_section_header
from utilities.data_handlers import EducationalDataLoader, ResultsAnalyzer

# Configuración
warnings.filterwarnings('ignore')

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizar uso de memoria del DataFrame."""
    print("Optimizando uso de memoria...")
    
    # Convertir enteros a tipos más pequeños
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    # Convertir flotantes a tipos más eficientes
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convertir strings repetitivos a categorías
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

def load_data_optimized(db: DatabaseManager, max_rows: int, strategy: str = "random") -> pd.DataFrame:
    """Carga optimizada de datos con diferentes estrategias."""
    print(f"Estrategia de carga: {strategy}")
    
    if strategy == "balanced":
        return load_balanced_sample(db, max_rows)
    elif strategy == "recent":
        return load_recent_data(db, max_rows)
    else:
        return load_random_sample(db, max_rows)

def load_random_sample(db: DatabaseManager, max_rows: int) -> pd.DataFrame:
    """Carga con muestreo aleatorio optimizado."""
    if max_rows:
        query = f"""
        SELECT v.*, i.final_result, i.gender, i.age_band, i.highest_education
        FROM studentVle v 
        JOIN studentInfo i ON v.id_student = i.id_student 
            AND v.code_presentation = i.code_presentation
        ORDER BY RAND() 
        LIMIT {max_rows}
        """
    else:
        query = """
        SELECT v.*, i.final_result, i.gender, i.age_band, i.highest_education
        FROM studentVle v 
        JOIN studentInfo i ON v.id_student = i.id_student 
            AND v.code_presentation = i.code_presentation
        """
    
    return execute_custom_query(query, db)

def load_balanced_sample(db: DatabaseManager, max_rows: int) -> pd.DataFrame:
    """Carga con muestreo balanceado por final_result."""
    print("Aplicando muestreo balanceado...")
    
    # Obtener distribución de clases
    query_dist = """
    SELECT final_result, COUNT(*) as count
    FROM studentInfo
    GROUP BY final_result
    """
    
    dist_df = execute_custom_query(query_dist, db)
    print("Distribución original:", dict(zip(dist_df['final_result'], dist_df['count'])))
    
    if max_rows:
        samples_per_class = max_rows // len(dist_df)
        dfs = []
        
        for _, row in dist_df.iterrows():
            result = row['final_result']
            query = f"""
            SELECT v.*, i.final_result, i.gender, i.age_band, i.highest_education
            FROM studentVle v 
            JOIN studentInfo i ON v.id_student = i.id_student 
                AND v.code_presentation = i.code_presentation
            WHERE i.final_result = '{result}'
            ORDER BY RAND() 
            LIMIT {samples_per_class}
            """
            df_class = execute_custom_query(query, db)
            dfs.append(df_class)
            print(f"   {result}: {len(df_class):,} registros")
        
        return pd.concat(dfs, ignore_index=True)
    else:
        return load_random_sample(db, None)

def load_recent_data(db: DatabaseManager, max_rows: int) -> pd.DataFrame:
    """Carga datos más recientes."""
    query = f"""
    SELECT v.*, i.final_result, i.gender, i.age_band, i.highest_education
    FROM studentVle v 
    JOIN studentInfo i ON v.id_student = i.id_student 
        AND v.code_presentation = i.code_presentation
    ORDER BY v.date DESC
    {f'LIMIT {max_rows}' if max_rows else ''}
    """
    
    return execute_custom_query(query, db)

def feature_selection_optimized(X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
    """Selección optimizada de features."""
    if X.shape[1] <= k:
        print(f"Número de features ({X.shape[1]}) menor al límite ({k}). No se requiere selección.")
        return X
    
    print(f"Seleccionando {k} features más relevantes de {X.shape[1]} disponibles...")
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Features seleccionadas: {list(selected_features)[:10]}...")
    return X_selected_df

def train_comprehensive_models(X: pd.DataFrame, y: pd.Series, output_dir: str) -> Dict[str, Any]:
    """
    Entrena múltiples modelos supervisados con evaluación completa.
    Mínimo 5 algoritmos requeridos para el proyecto.
    """
    print("\nENTRENAMIENTO COMPREHENSIVO DE MODELOS")
    print("=" * 50)
    
    # Preparar datos
    X_clean, y_clean = prepare_data_for_modeling(X, y)
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Definir modelos (mínimo 5 requeridos)
    models_config = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'NeuralNetwork': MLPClassifier(random_state=42, max_iter=500)
    }
    
    all_results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models_config.items():
        print(f"\nEntrenando {name}...")
        
        # Pipeline con SMOTE y escalado
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Entrenar
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predicciones
        y_pred = pipeline.predict(X_test)
        y_pred_proba = None
        try:
            y_pred_proba = pipeline.predict_proba(X_test)
        except:
            pass
        
        # Calcular métricas completas
        metrics = calculate_complete_metrics(y_test, y_pred, y_pred_proba)
        
        # Guardar modelo y resultados
        model_filename = f"{output_dir}/model_{name.lower()}.joblib"
        joblib.dump(pipeline, model_filename)
        
        all_results[name] = {
            'model': pipeline,
            'metrics': metrics,
            'predictions': {'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba},
            'train_time': train_time
        }
        
        # Tracking del mejor modelo
        if metrics['f1_weighted'] > best_score:
            best_score = metrics['f1_weighted']
            best_model = pipeline
            best_model_name = name
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.3f}")
        print(f"  Tiempo: {train_time:.1f}s")
    
    # Generar reportes comparativos
    generate_model_comparison_report(all_results, output_dir)
    
    print(f"\nMejor modelo: {best_model_name} (F1: {best_score:.3f})")
    
    return all_results

def prepare_data_for_modeling(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara los datos para modelado."""
    # Solo variables numéricas
    X_numeric = X.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] == 0:
        # Convertir categóricas a dummies si no hay numéricas
        X_numeric = pd.get_dummies(X, drop_first=True)
    
    # Manejar valores nulos
    X_clean = X_numeric.fillna(X_numeric.median())
    
    # Codificar variable objetivo si es categórica
    if y.dtype == 'object':
        le = LabelEncoder()
        y_clean = pd.Series(le.fit_transform(y), index=y.index)
    else:
        y_clean = y
    
    # Alinear índices
    common_idx = X_clean.index.intersection(y_clean.index)
    X_clean = X_clean.loc[common_idx]
    y_clean = y_clean.loc[common_idx]
    
    print(f"Datos preparados: {X_clean.shape[0]} registros, {X_clean.shape[1]} características")
    
    return X_clean, y_clean

def calculate_complete_metrics(y_test: pd.Series, y_pred: np.ndarray, 
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula todas las métricas requeridas por el proyecto.
    """
    metrics = {}
    
    # Métricas de clasificación básicas
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Métricas macro (promedio no ponderado)
    metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Métricas weighted (promedio ponderado)
    metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC si es posible
    try:
        if y_pred_proba is not None:
            if len(np.unique(y_test)) == 2:  # Binario
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:  # Multiclase
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            metrics['roc_auc'] = np.nan
    except:
        metrics['roc_auc'] = np.nan
    
    # Métricas de regresión (tratando predicciones como continuas)
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['r2'] = r2_score(y_test, y_pred)
    
    # Métricas manuales de matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:  # Matriz 2x2 (binario)
        tn, fp, fn, tp = cm.ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)
        
        # Cálculo manual de F1
        precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_manual'] = (2 * precision_manual * recall_manual / 
                              (precision_manual + recall_manual)) if (precision_manual + recall_manual) > 0 else 0
    
    return metrics

def generate_model_comparison_report(all_results: Dict[str, Any], output_dir: str):
    """Genera reporte comparativo de todos los modelos."""
    comparison_data = []
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1_Macro': metrics['f1_macro'],
            'F1_Weighted': metrics['f1_weighted'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'ROC_AUC': metrics['roc_auc'],
            'MSE': metrics['mse'],
            'R2': metrics['r2'],
            'Train_Time': results['train_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1_Weighted', ascending=False)
    
    # Guardar reporte
    comparison_df.to_csv(f"{output_dir}/model_comparison_report.csv", index=False)
    
    print("\nREPORTE COMPARATIVO DE MODELOS:")
    print("=" * 60)
    print(comparison_df.round(4).to_string(index=False, max_colwidth=15))

def main_improved(max_rows: int = 500_000, strategy: str = "balanced", 
                 quick_mode: bool = False, test_hypotheses: bool = True):
    """
    Pipeline principal mejorado que cumple con todos los requisitos del proyecto.
    """
    
    # Configuración
    results_dir = "results"
    output_dir = "results_final"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PIPELINE OULAD MEJORADO - PROYECTO COLABORATIVO")
    print("=" * 80)
    print(f"Configuración:")
    print(f"  Max registros: {max_rows:,}")
    print(f"  Estrategia: {strategy}")
    print(f"  Modo rápido: {quick_mode}")
    print(f"  Prueba de hipótesis: {test_hypotheses}")
    
    start_time = time.time()
    
    try:
        # 1. CONEXIÓN A BASE DE DATOS
        print(f"\n{color_azul_destacado}FASE 1: CONEXIÓN A BASE DE DATOS{color_reset}")
        db = DatabaseManager()
        db.test_connection()
        
        # 2. CARGA OPTIMIZADA DE DATOS
        print(f"\n{color_azul_destacado}FASE 2: CARGA OPTIMIZADA DE DATOS{color_reset}")
        
        if quick_mode and max_rows:
            # Usar carga optimizada
            df_merged = load_data_optimized(db, max_rows, strategy)
            df_merged = optimize_memory_usage(df_merged)
        else:
            # Usar método original pero optimizado
            print("Vinculando datos...")
            for i in tqdm(range(0, 101, 10), desc="Vinculando datos", ncols=80):
                time.sleep(0.5)  # Reducido para ser más rápido
            
            df_merged = load_data_optimized(db, max_rows, strategy)
            df_merged = optimize_memory_usage(df_merged)
        
        print(f"Datos cargados: {len(df_merged):,} registros")
        
        # 3. PREPROCESAMIENTO
        print(f"\n{color_azul_destacado}FASE 3: PREPROCESAMIENTO Y FEATURE ENGINEERING{color_reset}")
        data_processor = DataProcessor()
        
        df_clean, num_vars = data_processor.clean_dataset(df_merged)
        df_feat = data_processor.create_features(df_clean)
        
        print(f"Datos procesados: {df_feat.shape}")
        
        # Liberar memoria
        del df_merged, df_clean
        gc.collect()
        
        # 4. ANÁLISIS EXPLORATORIO MEJORADO
        print(f"\n{color_azul_destacado}FASE 4: ANÁLISIS EXPLORATORIO MEJORADO{color_reset}")
        data_explorer = DataExplorer()
        
        # EDA básico
        if quick_mode:
            print("EDA en modo rápido")
            sample_size = min(10000, len(df_feat))
            df_sample = df_feat.sample(n=sample_size, random_state=42) if len(df_feat) > sample_size else df_feat
            
            data_explorer.single_variable_analysis(df_sample, output_dir=results_dir)
            data_explorer.dual_variable_analysis(df_sample, output_dir=results_dir)
        else:
            # EDA completo
            if hasattr(data_explorer, 'comprehensive_eda_analysis'):
                data_explorer.comprehensive_eda_analysis(df_feat, results_dir)
            else:
                data_explorer.single_variable_analysis(df_feat, output_dir=results_dir)
                data_explorer.dual_variable_analysis(df_feat, output_dir=results_dir)
                data_explorer.generate_boxplots(df_feat, output_dir=results_dir)
        
        # Prueba de hipótesis si está disponible
        if test_hypotheses and hasattr(data_explorer, 'test_project_hypotheses'):
            hypothesis_results = data_explorer.test_project_hypotheses(df_feat, output_dir)
        
        # 5. ENTRENAMIENTO DE MODELOS COMPREHENSIVO
        print(f"\n{color_azul_destacado}FASE 5: ENTRENAMIENTO DE MODELOS{color_reset}")
        
        # Preparar datos para modelado
        if 'final_result' in df_feat.columns:
            X = df_feat.drop(columns=['final_result'])
            y = df_feat['final_result']
            
            # Feature selection si es necesario
            if X.select_dtypes(include=[np.number]).shape[1] > 20:
                X_numeric = X.select_dtypes(include=[np.number])
                if len(X_numeric.columns) > 0:
                    X_selected = feature_selection_optimized(X_numeric, y, k=20)
                    # Combinar con variables categóricas importantes
                    X_categorical = X.select_dtypes(include=['object', 'category'])
                    if len(X_categorical.columns) > 0:
                        X = pd.concat([X_selected, X_categorical], axis=1)
                    else:
                        X = X_selected
            
            # Entrenamiento comprehensivo
            model_results = train_comprehensive_models(X, y, output_dir)
            
            # Análisis de resultados con ResultsAnalyzer mejorado
            results_analyzer = ResultsAnalyzer()
            
            # Obtener mejores predicciones
            best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k]['metrics']['f1_weighted'])
            best_predictions = model_results[best_model_name]['predictions']
            
            # Guardar métricas completas usando el método mejorado
            if hasattr(results_analyzer, 'save_complete_performance_metrics'):
                results_analyzer.save_complete_performance_metrics(
                    best_predictions['y_test'],
                    best_predictions['y_pred'],
                    best_predictions.get('y_pred_proba'),
                    path=output_dir
                )
            else:
                # Usar método original como fallback
                results_analyzer.save_performance_metrics(
                    best_predictions['y_test'],
                    best_predictions['y_pred'],
                    output_dir
                )
            
            # Crear visualizaciones
            results_analyzer.create_confusion_matrix_plot(
                best_predictions['y_test'],
                best_predictions['y_pred'],
                output_dir
            )
            
            # Feature importance si está disponible
            best_model = model_results[best_model_name]['model']
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                results_analyzer.create_feature_importance_plot(
                    best_model.named_steps['classifier'],
                    X.columns,
                    output_dir
                )
            
        else:
            print("No se encontró variable objetivo 'final_result'")
            model_results = None
        
        # 6. REPORTE FINAL
        total_time = time.time() - start_time
        generate_final_project_report(df_feat, model_results, total_time, output_dir)
        
    except Exception as e:
        print(f"Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n{color_azul_destacado}PIPELINE COMPLETADO EXITOSAMENTE{color_reset}")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"Resultados guardados en: {output_dir}/")

def generate_final_project_report(df: pd.DataFrame, model_results: Optional[Dict] = None,
                                 execution_time: float = 0, output_dir: str = "results"):
    """Genera reporte final completo del proyecto."""
    
    report_data = {
        'total_records_processed': len(df),
        'total_features': df.shape[1],
        'execution_time_minutes': execution_time / 60,
        'missing_values_handled': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    if model_results:
        best_model = max(model_results.keys(), 
                        key=lambda k: model_results[k]['metrics']['f1_weighted'])
        
        report_data.update({
            'best_model': best_model,
            'best_f1_score': model_results[best_model]['metrics']['f1_weighted'],
            'best_accuracy': model_results[best_model]['metrics']['accuracy'],
            'models_trained': len(model_results),
            'algorithms_used': list(model_results.keys())
        })
    
    # Guardar reporte
    summary_df = pd.DataFrame([{
        'Metric': 'Records Processed',
        'Value': f"{len(df):,}"
    }, {
        'Metric': 'Features Generated', 
        'Value': df.shape[1]
    }, {
        'Metric': 'Execution Time (min)',
        'Value': f"{execution_time/60:.1f}"
    }, {
        'Metric': 'Best Model',
        'Value': report_data.get('best_model', 'N/A')
    }, {
        'Metric': 'Best F1-Score',
        'Value': f"{report_data.get('best_f1_score', 0):.3f}"
    }])
    
    summary_df.to_csv(f"{output_dir}/project_summary.csv", index=False)
    
    print("\nRESUMEN FINAL DEL PROYECTO")
    print("=" * 50)
    print(summary_df.to_string(index=False))

# Función principal compatible con tu estructura actual
def main(max_rows: int = 200_000, strategy: str = "random", quick_mode: bool = None):
    """Pipeline principal compatible con la estructura existente."""
    
    # Auto-detectar modo rápido basado en tamaño
    if quick_mode is None:
        quick_mode = max_rows > 100_000
    
    # Llamar a la función mejorada
    main_improved(
        max_rows=max_rows,
        strategy=strategy,
        quick_mode=quick_mode,
        test_hypotheses=True
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline OULAD mejorado para grandes volúmenes.")
    parser.add_argument('--max_rows', type=int, default=500_000, 
                       help='Número máximo de filas (default: 500000)')
    parser.add_argument('--strategy', choices=['random', 'balanced', 'recent'], 
                       default='balanced', help='Estrategia de muestreo')
    parser.add_argument('--quick', action='store_true', 
                       help='Modo rápido (menos análisis)')
    parser.add_argument('--full', action='store_true', 
                       help='Forzar modo completo')
    parser.add_argument('--no_hypotheses', action='store_true',
                       help='Omitir pruebas de hipótesis')
    
    args = parser.parse_args()
    
    # Determinar modo
    if args.full:
        quick_mode = False
    elif args.quick:
        quick_mode = True
    else:
        quick_mode = None  # Auto-detectar
    
    main_improved(
        max_rows=args.max_rows,
        strategy=args.strategy, 
        quick_mode=quick_mode,
        test_hypotheses=not args.no_hypotheses
    )