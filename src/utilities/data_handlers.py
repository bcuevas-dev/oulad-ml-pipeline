import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_score, 
                           recall_score, mean_squared_error, r2_score, roc_auc_score, 
                           classification_report)

class EducationalDataLoader:
    """Clase para cargar los datos educativos y datasets complementarios."""
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_virtual_environment_data(self, filename: str = "Full_vle_train.csv") -> pd.DataFrame:
        path = f"{self.base_path}/{filename}"
        return pd.read_csv(path, encoding='latin1')

    def load_assessment_data(self, filename: str = "Full_assess_train.csv") -> pd.DataFrame:
        path = f"{self.base_path}/{filename}"
        return pd.read_csv(path, encoding='latin1')

class ResultsAnalyzer:
    """Clase mejorada para interpretación de resultados y métricas."""
    def __init__(self):
        pass

    def save_performance_metrics(self, y_test, y_pred, path: str):
        """Método original mantenido para compatibilidad."""
        # Exportar y_test, y_pred y métricas manuales a CSV
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        df.to_csv(os.path.join(path, 'test_predictions.csv'), index=False)
        # Cálculo manual de métricas
        confusion_mat = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = (confusion_mat.ravel() if confusion_mat.size == 4 else (0,0,0,0))
        # Detectar si el problema es binario o multiclase
        num_classes = len(np.unique(y_test))
        if num_classes == 2:
            avg_type = 'binary'
        else:
            avg_type = 'macro'
        f1 = f1_score(y_test, y_pred, average=avg_type)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = pd.DataFrame([{'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'f1_score': f1, 'accuracy': acc, 'precision': prec, 'recall': rec, 'mse': mse, 'r2': r2}])
        metrics.to_csv(os.path.join(path, 'performance_metrics.csv'), index=False)
        return confusion_mat, metrics

    def save_complete_performance_metrics(self, y_test, y_pred, y_pred_proba=None, 
                                        y_research_test=None, y_research_pred=None, 
                                        path: str = "results"):
        """
        MÉTODO NUEVO: Guarda todas las métricas requeridas por el proyecto.
        
        Métricas requeridas por el proyecto:
        - precision_macro, recall_macro, f1_macro, accuracy, roc_auc
        - mse, r2, msePI2, r2PI2
        - Cálculo manual de f1_score con TP, FP, TN, FN
        """
        
        # Guardar predicciones caso a caso (y_test, y_pred)
        predictions_df = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred
        })
        predictions_df.to_csv(os.path.join(path, 'test_predictions_complete.csv'), index=False)
        
        # Matriz de confusión
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        # Métricas básicas requeridas
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Métricas weighted adicionales
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC si es posible calcular
        roc_auc = None
        try:
            if y_pred_proba is not None:
                if len(np.unique(y_test)) == 2:  # Clasificación binaria
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:  # Clasificación multiclase
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                # Intentar calcular con predicciones discretas (menos preciso)
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred)
        except Exception as e:
            print(f"No se pudo calcular ROC-AUC: {e}")
            roc_auc = None
        
        # Métricas de regresión
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Métricas de investigación (PI2) si están disponibles
        msePI2 = None
        r2PI2 = None
        if y_research_test is not None and y_research_pred is not None:
            msePI2 = mean_squared_error(y_research_test, y_research_pred)
            r2PI2 = r2_score(y_research_test, y_research_pred)
        
        # Cálculo manual de métricas de matriz de confusión
        tp = fp = tn = fn = 0
        f1_manual = 0
        
        if confusion_mat.size == 4:  # Clasificación binaria
            tn, fp, fn, tp = confusion_mat.ravel()
            
            # Cálculo manual de F1-Score
            precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_manual = (2 * precision_manual * recall_manual / 
                        (precision_manual + recall_manual)) if (precision_manual + recall_manual) > 0 else 0
        elif confusion_mat.size > 4:  # Clasificación multiclase
            # Para multiclase, sumar TP, FP, etc. de todas las clases
            tp = np.diag(confusion_mat).sum()
            fp = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
            fn = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
            tn = confusion_mat.sum() - (fp.sum() + fn.sum() + tp)
            
            fp = fp.sum()
            fn = fn.sum()
        
        # Crear DataFrame con TODAS las métricas requeridas
        metrics_complete = pd.DataFrame([{
            # Matriz de confusión manual
            'TP': int(tp),
            'FP': int(fp), 
            'TN': int(tn),
            'FN': int(fn),
            'f1_score_manual': f1_manual,
            
            # Métricas requeridas por el proyecto
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            
            # Métricas adicionales weighted
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            
            # Métricas de regresión
            'mse': mse,
            'r2': r2,
            
            # Métricas de investigación (PI2)
            'msePI2': msePI2,
            'r2PI2': r2PI2
        }])
        
        # Guardar métricas completas
        metrics_complete.to_csv(os.path.join(path, 'performance_metrics_complete.csv'), index=False)
        
        # Reporte detallado en consola
        print("\n" + "="*60)
        print("MÉTRICAS DE RENDIMIENTO COMPLETAS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        if confusion_mat.size == 4:
            print(f"\nMétricas de Matriz de Confusión (Binaria):")
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            print(f"F1-Score Manual: {f1_manual:.4f}")
        
        if msePI2 is not None:
            print(f"\nMétricas de Investigación (PI2):")
            print(f"MSE PI2: {msePI2:.4f}")
            print(f"R² PI2: {r2PI2:.4f}")
        
        print("="*60)
        
        return confusion_mat, metrics_complete

    def create_confusion_matrix_plot(self, y_test, y_pred, path: str):
        """Método original mantenido para compatibilidad."""
        confusion_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.savefig(os.path.join(path, 'confusion_matrix.png'))
        plt.close()

    def create_advanced_confusion_matrix_plot(self, y_test, y_pred, path: str):
        """MÉTODO NUEVO: Matriz de confusión mejorada y detallada."""
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Obtener etiquetas únicas
        unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        
        # Crear heatmap mejorado
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels,
                   cbar_kws={'label': 'Número de Predicciones'})
        plt.title('Matriz de Confusión Detallada', fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        
        # Añadir estadísticas en el gráfico
        accuracy = accuracy_score(y_test, y_pred)
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'confusion_matrix_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_feature_importance_plot(self, model, feature_names, path: str):
        """
        Método CORREGIDO de importancia de características con manejo robusto de errores.
        VERSIÓN COMPLETA que soluciona el error "All arrays must be of the same length"
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # CORRECCIÓN PRINCIPAL: Verificar y ajustar longitudes
            print(f"Debug - Importancias del modelo: {len(importances)}")
            print(f"Debug - Nombres de características: {len(feature_names)}")
            
            if len(importances) != len(feature_names):
                print(f"ADVERTENCIA: Longitudes no coinciden. Ajustando...")
                
                # Estrategia 1: Tomar el mínimo común
                min_length = min(len(importances), len(feature_names))
                importances_adj = importances[:min_length]
                feature_names_adj = list(feature_names)[:min_length]
                
                print(f"Ajustado a longitud común: {min_length}")
                
                # Si aún hay discrepancia, crear nombres genéricos
                if len(importances_adj) != len(feature_names_adj):
                    if len(importances_adj) > len(feature_names_adj):
                        # Más importancias que nombres
                        additional_names = [f'feature_{i}' for i in range(len(feature_names_adj), len(importances_adj))]
                        feature_names_adj.extend(additional_names)
                    else:
                        # Más nombres que importancias
                        feature_names_adj = feature_names_adj[:len(importances_adj)]
            else:
                importances_adj = importances
                feature_names_adj = list(feature_names)
            
            # Verificación final de seguridad
            try:
                assert len(importances_adj) == len(feature_names_adj), \
                    f"Error crítico: {len(importances_adj)} != {len(feature_names_adj)}"
            except AssertionError as e:
                print(f"Error de alineación: {e}")
                # Crear versión de emergencia
                return self._emergency_feature_importance(model, feature_names, path)
            
            try:
                indices = np.argsort(importances_adj)[::-1]
                
                # Gráfico original (mantenido para compatibilidad)
                plt.figure(figsize=(10,6))
                plt.title('Importancia de variables')
                plt.bar(range(len(importances_adj)), importances_adj[indices], align='center')
                
                # Limitar etiquetas para evitar sobrecarga visual
                max_labels = min(20, len(importances_adj))
                tick_positions = range(max_labels)
                tick_labels = [feature_names_adj[indices[i]] for i in range(max_labels)]
                
                plt.xticks(tick_positions, tick_labels, rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(path, 'feature_importance.png'))
                plt.close()
                
                # Gráfico mejorado - Top características (versión segura)
                top_n = min(15, len(importances_adj))
                plt.figure(figsize=(12, 8))
                
                top_importances = importances_adj[indices[:top_n]]
                top_features = [feature_names_adj[indices[i]] for i in range(top_n)]
                
                # Gráfico horizontal para mejor legibilidad
                y_pos = np.arange(top_n)
                plt.barh(y_pos, top_importances, align='center', color='skyblue', edgecolor='navy')
                plt.yticks(y_pos, top_features)
                plt.xlabel('Importancia', fontsize=12)
                plt.title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()  # Para mostrar la más importante arriba
                
                # Añadir valores en las barras
                for i, v in enumerate(top_importances):
                    if v > 0:  # Solo mostrar valores positivos
                        plt.text(v + max(top_importances)*0.01, i, f'{v:.3f}', 
                                va='center', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(os.path.join(path, 'feature_importance_top15.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Guardar importancias en CSV - VERSIÓN ULTRA SEGURA
                importance_data = []
                for i in range(len(feature_names_adj)):
                    importance_data.append({
                        'feature': feature_names_adj[i],
                        'importance': float(importances_adj[i]),  # Convertir a float nativo
                        'rank': int(np.where(indices == i)[0][0] + 1)  # Ranking
                    })
                
                importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
                importance_df.to_csv(os.path.join(path, 'feature_importances.csv'), index=False)
                
                print(f"\nTop 10 características más importantes:")
                for i, row in enumerate(importance_df.head(10).itertuples(index=False)):
                    print(f"  {i+1:2d}. {row.feature}: {row.importance:.4f}")
                
                return True
                    
            except Exception as e:
                print(f"Error en gráficos avanzados: {e}")
                print("Creando versión de respaldo ultra-simple...")
                return self._backup_feature_importance(importances_adj, feature_names_adj, path)
        else:
            print("El modelo no tiene atributo 'feature_importances_'")
            return False

    def _backup_feature_importance(self, importances, feature_names, path: str):
        """Versión de respaldo para feature importance."""
        try:
            backup_data = []
            for i, (feat, imp) in enumerate(zip(feature_names, importances)):
                backup_data.append({
                    'feature_index': i,
                    'feature_name': str(feat),
                    'importance': float(imp)
                })
            
            backup_df = pd.DataFrame(backup_data).sort_values('importance', ascending=False)
            backup_df.to_csv(os.path.join(path, 'feature_importance_backup.csv'), index=False)
            
            # Gráfico mínimo
            plt.figure(figsize=(8, 6))
            top_5 = backup_df.head(5)
            plt.bar(range(len(top_5)), top_5['importance'])
            plt.title('Top 5 Características Más Importantes (Respaldo)')
            plt.xlabel('Ranking')
            plt.ylabel('Importancia')
            plt.xticks(range(len(top_5)), [f'#{i+1}' for i in range(len(top_5))])
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'feature_importance_minimal.png'))
            plt.close()
            
            print("Versión de respaldo creada exitosamente")
            return True
            
        except Exception as e2:
            print(f"Error crítico en respaldo: {e2}")
            return False

    def _emergency_feature_importance(self, model, feature_names, path: str):
        """Función de emergencia para casos extremos."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Crear diccionario simple
                importance_dict = {}
                for i, imp in enumerate(importances):
                    key = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                    importance_dict[key] = float(imp)
                
                # Ordenar por importancia
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                # Guardar en archivo de texto
                with open(f"{path}/feature_importance_emergency.txt", 'w') as f:
                    f.write("FEATURE IMPORTANCE REPORT (Emergency Mode)\n")
                    f.write("=" * 50 + "\n\n")
                    for i, (feature, importance) in enumerate(sorted_features[:20]):
                        f.write(f"{i+1:2d}. {feature}: {importance:.4f}\n")
                
                print("Reporte de emergencia de feature importance creado")
                return True
        except Exception as e:
            print(f"Falló incluso el método de emergencia: {e}")
            return False

    def create_advanced_visualizations(self, y_test, y_pred, y_pred_proba=None, 
                                     feature_names=None, model=None, path="results"):
        """MÉTODO NUEVO: Crea visualizaciones avanzadas requeridas por el proyecto."""
        
        # 1. Matriz de confusión detallada
        self.create_advanced_confusion_matrix_plot(y_test, y_pred, path)
        
        # 2. Scatter plots de predicciones vs reales
        plt.figure(figsize=(15, 5))
        
        # Scatter plot principal
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea perfecta')
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs Valores Reales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico de residuales
        plt.subplot(1, 3, 2)
        residuals = np.array(y_test) - np.array(y_pred)
        plt.scatter(y_pred, residuals, alpha=0.6, color='red', edgecolors='black', linewidth=0.5)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.8)
        plt.xlabel('Predicciones')
        plt.ylabel('Residuales')
        plt.title('Gráfico de Residuales')
        plt.grid(True, alpha=0.3)
        
        # Histograma de residuales
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Residuales')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Residuales')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'scatter_plots_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribución de probabilidades por clase (si están disponibles)
        if y_pred_proba is not None and len(np.unique(y_test)) <= 10:
            plt.figure(figsize=(12, 6))
            
            unique_classes = np.unique(y_test)
            for i, class_label in enumerate(unique_classes):
                if i < y_pred_proba.shape[1]:
                    class_indices = np.array(y_test) == class_label
                    class_probs = y_pred_proba[class_indices, i]
                    plt.hist(class_probs, alpha=0.6, label=f'Clase {class_label}', bins=20)
            
            plt.xlabel('Probabilidad Predicha')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Probabilidades por Clase')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Feature importance si está disponible
        if model is not None and hasattr(model, 'feature_importances_') and feature_names is not None:
            self.create_feature_importance_plot(model, feature_names, path)
        
        # 5. Reporte de clasificación detallado
        try:
            class_report = classification_report(y_test, y_pred, output_dict=True)
            class_report_df = pd.DataFrame(class_report).transpose()
            class_report_df.to_csv(os.path.join(path, 'classification_report_detailed.csv'))
            
            print("\nReporte de Clasificación Detallado:")
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print(f"No se pudo generar el reporte de clasificación: {e}")

    def generate_model_comparison_report(self, model_results: dict, path: str):
        """MÉTODO NUEVO: Genera reporte comparativo de múltiples modelos."""
        
        if not model_results:
            print("No hay resultados de modelos para comparar.")
            return
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1_Macro': metrics.get('f1_macro', 0),
                    'F1_Weighted': metrics.get('f1_weighted', 0),
                    'Precision_Macro': metrics.get('precision_macro', 0),
                    'Recall_Macro': metrics.get('recall_macro', 0),
                    'ROC_AUC': metrics.get('roc_auc', 0),
                    'MSE': metrics.get('mse', 0),
                    'R2': metrics.get('r2', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1_Weighted', ascending=False)
            
            # Guardar reporte
            comparison_df.to_csv(os.path.join(path, 'model_comparison_report.csv'), index=False)
            
            # Crear visualización comparativa
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy
            comparison_df.plot(x='Model', y='Accuracy', kind='bar', ax=axes[0,0], 
                             color='skyblue', legend=False)
            axes[0,0].set_title('Accuracy por Modelo')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # F1 Score
            comparison_df.plot(x='Model', y='F1_Weighted', kind='bar', ax=axes[0,1], 
                             color='lightgreen', legend=False)
            axes[0,1].set_title('F1-Score Weighted por Modelo')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # ROC-AUC (si está disponible)
            roc_data = comparison_df[comparison_df['ROC_AUC'] > 0]
            if not roc_data.empty:
                roc_data.plot(x='Model', y='ROC_AUC', kind='bar', ax=axes[1,0], 
                            color='orange', legend=False)
                axes[1,0].set_title('ROC-AUC por Modelo')
                axes[1,0].set_ylabel('ROC-AUC')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # R2 Score
            comparison_df.plot(x='Model', y='R2', kind='bar', ax=axes[1,1], 
                             color='salmon', legend=False)
            axes[1,1].set_title('R² Score por Modelo')
            axes[1,1].set_ylabel('R²')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'model_metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nREPORTE COMPARATIVO DE MODELOS:")
            print("="*60)
            print(comparison_df.round(4).to_string(index=False))
            
            return comparison_df
        
        return None

    # MÉTODO ADICIONAL: Función utilitaria para debugging
    def debug_feature_importance(self, model, feature_names, path: str):
        """Función de debugging para entender problemas de feature importance."""
        print("\n" + "="*60)
        print("DEBUG FEATURE IMPORTANCE")
        print("="*60)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            print(f"Tipo de importances: {type(importances)}")
            print(f"Shape de importances: {importances.shape}")
            print(f"Longitud de importances: {len(importances)}")
            print(f"Primeros 5 valores: {importances[:5]}")
        else:
            print("El modelo NO tiene atributo 'feature_importances_'")
            return
        
        print(f"\nTipo de feature_names: {type(feature_names)}")
        print(f"Longitud de feature_names: {len(feature_names)}")
        if hasattr(feature_names, 'shape'):
            print(f"Shape de feature_names: {feature_names.shape}")
        print(f"Primeros 5 nombres: {list(feature_names)[:5]}")
        
        # Verificar discrepancias
        diff = len(importances) - len(feature_names)
        print(f"\nDiferencia de longitudes: {diff}")
        
        if diff != 0:
            print("PROBLEMA DETECTADO: Las longitudes no coinciden")
            print("Estrategias de corrección:")
            print("1. Usar longitud mínima")
            print("2. Generar nombres adicionales")
            print("3. Truncar importances")
        else:
            print("✓ Las longitudes coinciden correctamente")
        
        # Guardar debug info
        debug_info = {
            'importances_length': len(importances),
            'feature_names_length': len(feature_names),
            'difference': diff,
            'importances_type': str(type(importances)),
            'feature_names_type': str(type(feature_names))
        }
        
        debug_df = pd.DataFrame([debug_info])
        debug_df.to_csv(os.path.join(path, 'feature_importance_debug.csv'), index=False)
        print(f"\nInfo de debug guardada en: {path}/feature_importance_debug.csv")

    # MÉTODO PARA VALIDAR DATOS ANTES DEL ANÁLISIS
    def validate_model_data(self, y_test, y_pred, y_pred_proba=None):
        """Valida los datos antes de realizar análisis."""
        print("\n" + "="*50)
        print("VALIDACIÓN DE DATOS DEL MODELO")
        print("="*50)
        
        # Validar y_test
        print(f"y_test - Tipo: {type(y_test)}, Longitud: {len(y_test)}")
        if hasattr(y_test, 'dtype'):
            print(f"y_test - Dtype: {y_test.dtype}")
        print(f"y_test - Valores únicos: {len(np.unique(y_test))}")
        print(f"y_test - Primeros 5: {list(y_test)[:5]}")
        
        # Validar y_pred
        print(f"\ny_pred - Tipo: {type(y_pred)}, Longitud: {len(y_pred)}")
        if hasattr(y_pred, 'dtype'):
            print(f"y_pred - Dtype: {y_pred.dtype}")
        print(f"y_pred - Valores únicos: {len(np.unique(y_pred))}")
        print(f"y_pred - Primeros 5: {list(y_pred)[:5]}")
        
        # Validar y_pred_proba si existe
        if y_pred_proba is not None:
            print(f"\ny_pred_proba - Tipo: {type(y_pred_proba)}")
            if hasattr(y_pred_proba, 'shape'):
                print(f"y_pred_proba - Shape: {y_pred_proba.shape}")
            print(f"y_pred_proba - Primeras 3 filas:\n{y_pred_proba[:3]}")
        
        # Verificar alineación
        length_match = len(y_test) == len(y_pred)
        print(f"\n¿Longitudes de y_test y y_pred coinciden? {length_match}")
        
        if not length_match:
            print("⚠️  ADVERTENCIA: Las longitudes no coinciden")
            return False
        
        # Verificar valores nulos
        y_test_nulls = pd.Series(y_test).isnull().sum()
        y_pred_nulls = pd.Series(y_pred).isnull().sum()
        
        print(f"Valores nulos en y_test: {y_test_nulls}")
        print(f"Valores nulos en y_pred: {y_pred_nulls}")
        
        if y_test_nulls > 0 or y_pred_nulls > 0:
            print("ADVERTENCIA: Se encontraron valores nulos")
            return False
        
        print(" Validación completada exitosamente")
        return True