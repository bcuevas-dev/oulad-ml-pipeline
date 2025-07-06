import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings

class DataProcessor:
    """Clase mejorada para limpieza y preprocesamiento de datos educativos."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.feature_stats = {}
    
    def limit_outliers(self, column: pd.Series, method: str = "iqr", percentile: float = 0.98) -> pd.Series:
        """
        Limita los valores atípicos usando diferentes métodos.
        
        Args:
            column: Serie de datos
            method: 'percentile', 'iqr', 'zscore'
            percentile: Percentil a usar si method='percentile'
        """
        if column.empty:
            return column
        
        # Asegurar que sea numérico
        column = pd.to_numeric(column, errors='coerce').fillna(0)
        
        if method == "percentile":
            # Método original mejorado
            upper_bound = column.quantile(percentile)
            lower_bound = column.quantile(1 - percentile)
            return column.clip(lower=lower_bound, upper=upper_bound)
        
        elif method == "iqr":
            # Método de Rango Intercuartílico
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Guardar bounds para reporte
            self.outlier_bounds[column.name] = {
                'method': 'iqr',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': len(column[(column < lower_bound) | (column > upper_bound)])
            }
            
            return column.clip(lower=lower_bound, upper=upper_bound)
        
        elif method == "zscore":
            # Método Z-Score
            try:
                z_scores = np.abs(stats.zscore(column))
                threshold = 3  # 3 desviaciones estándar
                outlier_mask = z_scores > threshold
                
                self.outlier_bounds[column.name] = {
                    'method': 'zscore',
                    'threshold': threshold,
                    'outliers_count': outlier_mask.sum()
                }
                
                # Reemplazar outliers con percentil 95/5
                upper_replacement = column.quantile(0.95)
                lower_replacement = column.quantile(0.05)
                
                column_clean = column.copy()
                column_clean[outlier_mask & (column > column.median())] = upper_replacement
                column_clean[outlier_mask & (column <= column.median())] = lower_replacement
                
                return column_clean
            except:
                # Si falla z-score, usar método IQR como fallback
                return self.limit_outliers(column, method="iqr")
        
        else:
            raise ValueError(f"Método '{method}' no reconocido. Use 'percentile', 'iqr', o 'zscore'")

    def clean_dataset(self, df: pd.DataFrame, requirement: int = 1, 
                     outlier_method: str = "iqr", advanced_cleaning: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Limpieza avanzada del dataset con múltiples opciones.
        
        Args:
            df: DataFrame a limpiar
            requirement: Nivel de requerimiento (1, 2, 3)
            outlier_method: Método para tratar outliers
            advanced_cleaning: Si aplicar limpieza avanzada
        """
        print("INICIANDO LIMPIEZA AVANZADA DEL DATASET")
        print("-" * 50)
        
        df = df.copy()
        initial_shape = df.shape
        cleaning_report = {}
        
        # ================================
        # CORRECCIÓN DEL ERROR CATEGÓRICO
        # ================================
        categorical_columns = df.select_dtypes(include=['category']).columns
        if len(categorical_columns) > 0:
            print(f"Convirtiendo {len(categorical_columns)} columnas categóricas a object...")
            for col in categorical_columns:
                df[col] = df[col].astype('object')
        
        # 1. ANÁLISIS INICIAL DE CALIDAD
        print("1. Análisis inicial de calidad de datos...")
        null_analysis = df.isnull().sum()
        duplicate_count = df.duplicated().sum()
        
        cleaning_report['initial_records'] = len(df)
        cleaning_report['initial_features'] = df.shape[1]
        cleaning_report['initial_nulls'] = null_analysis.sum()
        cleaning_report['initial_duplicates'] = duplicate_count
        
        print(f"   - Registros iniciales: {len(df):,}")
        print(f"   - Características iniciales: {df.shape[1]}")
        print(f"   - Valores nulos totales: {null_analysis.sum():,}")
        print(f"   - Duplicados: {duplicate_count:,}")
        
        # 2. ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS O INÚTILES
        print("\n2. Eliminando columnas problemáticas...")
        columns_to_drop = []
        
        # Columnas completamente nulas
        completely_null = [col for col, v in null_analysis.items() if v == len(df)]
        columns_to_drop.extend(completely_null)
        
        # Columnas con un solo valor único (no informativas)
        if advanced_cleaning:
            single_value_cols = [col for col in df.columns 
                               if df[col].nunique() <= 1 and col not in ['final_result']]
            columns_to_drop.extend(single_value_cols)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"   - Columnas eliminadas: {len(columns_to_drop)} ({columns_to_drop})")
        
        cleaning_report['columns_dropped'] = columns_to_drop
        
        # 3. MANEJO INTELIGENTE DE VALORES NULOS
        print("\n3. Manejo inteligente de valores nulos...")
        
        # Eliminar registros con nulls en variables críticas
        critical_columns = ['imd_band'] if 'imd_band' in df.columns else []
        
        # Para datos educativos, también considerar estas como críticas
        educational_critical = ['final_result', 'id_student', 'code_module']
        critical_columns.extend([col for col in educational_critical if col in df.columns])
        
        initial_records = len(df)
        for col in critical_columns:
            before_drop = len(df)
            df = df.dropna(subset=[col])
            dropped = before_drop - len(df)
            if dropped > 0:
                print(f"   - Eliminados {dropped:,} registros sin '{col}'")
        
        # IMPUTACIÓN INTELIGENTE POR TIPO DE DATOS
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        
        # Imputar numéricas con mediana (más robusto que media)
        for col in numeric_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                if null_count / len(df) < 0.5:  # Si menos del 50% son nulos
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    print(f"   - '{col}': {null_count} nulos imputados con mediana ({median_value:.2f})")
                else:
                    print(f"   - WARNING: '{col}' tiene {null_count/len(df)*100:.1f}% nulos - considerar eliminar")
        
        # Imputar categóricas con moda o valor por defecto
        for col in object_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                if null_count / len(df) < 0.5:
                    mode_value = df[col].mode()
                    fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    print(f"   - '{col}': {null_count} nulos imputados con moda ('{fill_value}')")
        
        cleaning_report['records_after_null_cleaning'] = len(df)
        
        # 4. ELIMINAR DUPLICADOS
        print("\n4. Eliminando duplicados...")
        before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before_dedup - len(df)
        if duplicates_removed > 0:
            print(f"   - Duplicados eliminados: {duplicates_removed:,}")
        
        cleaning_report['duplicates_removed'] = duplicates_removed
        
        # 5. TRATAMIENTO AVANZADO DE OUTLIERS
        print(f"\n5. Tratamiento de outliers (método: {outlier_method})...")
        
        # Variables específicas para datos educativos OULAD
        educational_numeric_vars = ['num_of_prev_attempts', 'studied_credits']
        pattern_vars = [col for col in df.columns if any(col.startswith(p) for p in ['n_day', 'avg_sum', 'sum_click'])]
        
        numerical_variables = list(set(educational_numeric_vars + pattern_vars))
        
        # Agregar 'score' si está disponible y se requiere
        if requirement == 3 and 'score' in df.columns:
            numerical_variables.append('score')
        
        # Aplicar tratamiento de outliers
        outliers_treated = 0
        for col in numerical_variables:
            if col in df.columns and col != 'score':  # No tocar score si es variable objetivo
                try:
                    original_col = df[col].copy()
                    df[col] = self.limit_outliers(df[col], method=outlier_method)
                    
                    # Contar cuántos valores fueron modificados
                    changes = (original_col != df[col]).sum()
                    if changes > 0:
                        outliers_treated += changes
                        print(f"   - '{col}': {changes} outliers tratados")
                except Exception as e:
                    print(f"   - Error tratando outliers en '{col}': {e}")
        
        if outliers_treated == 0:
            print("   - No se encontraron outliers significativos")
        else:
            print(f"   - Total de outliers tratados: {outliers_treated}")
        
        cleaning_report['outliers_treated'] = outliers_treated
        cleaning_report['numerical_variables'] = numerical_variables
        
        # 6. VALIDACIONES FINALES
        print("\n6. Validaciones finales...")
        
        # Verificar que no hay infinitos o NaN
        numeric_data = df.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            inf_count = np.isinf(numeric_data).sum().sum()
            if inf_count > 0:
                print(f"   - WARNING: {inf_count} valores infinitos encontrados")
                df = df.replace([np.inf, -np.inf], np.nan)
                # Rellenar infinitos con mediana
                for col in numeric_data.columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
        
        final_nulls = df.isnull().sum().sum()
        if final_nulls > 0:
            print(f"   - WARNING: {final_nulls} valores nulos restantes")
        
        # 7. REPORTE FINAL DE LIMPIEZA
        cleaning_report.update({
            'final_records': len(df),
            'final_features': df.shape[1],
            'final_nulls': final_nulls,
            'records_removed': initial_shape[0] - len(df),
            'features_removed': initial_shape[1] - df.shape[1],
            'data_quality_score': self._calculate_quality_score(df)
        })
        
        print(f"\nRESUMEN DE LIMPIEZA:")
        print(f"   - Registros: {initial_shape[0]:,} → {len(df):,} ({cleaning_report['records_removed']:,} eliminados)")
        print(f"   - Características: {initial_shape[1]} → {df.shape[1]} ({cleaning_report['features_removed']} eliminadas)")
        print(f"   - Calidad de datos: {cleaning_report['data_quality_score']:.1f}%")
        
        # Guardar reporte de limpieza
        self.cleaning_report = cleaning_report
        
        return df, numerical_variables
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calcula un score de calidad de datos (0-100)."""
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness = (total_cells - null_cells) / total_cells
        
        # Penalizar por duplicados
        uniqueness = len(df.drop_duplicates()) / len(df)
        
        # Score final
        quality_score = (completeness * 0.7 + uniqueness * 0.3) * 100
        return quality_score

    def create_features(self, df: pd.DataFrame, requirement: int = 1, 
                       advanced_features: bool = True) -> pd.DataFrame:
        """
        Feature engineering avanzado para datos educativos.
        
        Args:
            df: DataFrame procesado
            requirement: Nivel de requerimiento
            advanced_features: Si crear características avanzadas
        """
        print("\nINICIANDO FEATURE ENGINEERING AVANZADO")
        print("-" * 50)
        
        df = df.copy()
        initial_features = df.shape[1]
        
        # 1. CARACTERÍSTICAS TEMPORALES AVANZADAS
        print("1. Creando características temporales...")
        if 'code_presentation' in df.columns:
            try:
                # Asegurar que sea string antes de manipular
                df['code_presentation'] = df['code_presentation'].astype(str)
                
                # Características temporales básicas
                df['year'] = pd.to_numeric(df['code_presentation'].str.strip().str[0:4], errors='coerce').fillna(2014).astype(int)
                df['semester'] = df['code_presentation'].str.strip().str[-1]
                
                if advanced_features:
                    # Características temporales avanzadas
                    df['is_recent_year'] = (df['year'] >= df['year'].quantile(0.75)).astype(int)
                    df['semester_numeric'] = df['semester'].map({'B': 1, 'J': 2}).fillna(0)
                    
                    # Tendencia temporal (años desde el primer año)
                    min_year = df['year'].min()
                    df['years_since_start'] = df['year'] - min_year
                    
                    print(f"   - Características temporales creadas: year, semester, is_recent_year, years_since_start")
            except Exception as e:
                print(f"   - Error creando características temporales: {e}")
        
        # 2. CODIFICACIÓN INTELIGENTE DE VARIABLES
        print("2. Codificación inteligente de variables...")
        
        # Crear copia de resultado final antes de codificar
        if 'final_result' in df.columns:
            df['final_result_original'] = df['final_result'].astype(str)
            
            if advanced_features:
                try:
                    # Variables dummy para final_result
                    df['is_pass'] = (df['final_result'] == 'Pass').astype(int)
                    df['is_distinction'] = (df['final_result'] == 'Distinction').astype(int)
                    df['is_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)
                    df['is_fail'] = (df['final_result'] == 'Fail').astype(int)
                    
                    # Variable de éxito académico (Pass o Distinction)
                    df['academic_success'] = ((df['final_result'] == 'Pass') | 
                                            (df['final_result'] == 'Distinction')).astype(int)
                    print(f"   - Variables de resultado creadas: is_pass, is_distinction, academic_success, etc.")
                except Exception as e:
                    print(f"   - Error creando variables de resultado: {e}")
        
        # Codificación con LabelEncoder mejorado
        if requirement == 2:
            encoding_columns = ['final_result', 'age_band', 'imd_band', 'disability', 
                              'gender', 'region', 'highest_education', 'code_module', 
                              'assessment_type', 'semester']
        else:
            encoding_columns = ['final_result', 'age_band', 'imd_band', 'disability', 
                              'gender', 'region', 'highest_education', 'code_module', 'semester']
        
        encoded_count = 0
        for col in encoding_columns:
            if col in df.columns:
                try:
                    # CORRECCIÓN: Asegurar que sea string antes de codificar
                    df[col] = df[col].astype(str)
                    
                    # Guardar encoder para uso futuro
                    col_encoder = preprocessing.LabelEncoder()
                    df[col] = col_encoder.fit_transform(df[col])
                    self.encoders[col] = col_encoder
                    encoded_count += 1
                except Exception as e:
                    print(f"   - No se pudo codificar '{col}': {e}")
        
        print(f"   - Variables codificadas: {encoded_count}")
        
        # 3. CARACTERÍSTICAS DE INTERACCIÓN AVANZADAS
        print("3. Creando características de interacción...")
        
        interaction_features = 0
        
        # Característica original
        if 'total_n_days' in df.columns and 'avg_total_sum_clicks' in df.columns:
            try:
                df['overall_interaction_score'] = pd.to_numeric(df['total_n_days'], errors='coerce').fillna(0) * pd.to_numeric(df['avg_total_sum_clicks'], errors='coerce').fillna(0)
                interaction_features += 1
            except Exception as e:
                print(f"   - Error creando overall_interaction_score: {e}")
        
        if advanced_features:
            # Características de participación
            if 'sum_click' in df.columns:
                try:
                    # Asegurar que sum_click sea numérico
                    df['sum_click'] = pd.to_numeric(df['sum_click'], errors='coerce').fillna(0)
                    
                    # Nivel de participación (versión segura)
                    if df['sum_click'].max() > df['sum_click'].min():  # Solo si hay variación
                        df['participation_level'] = pd.cut(df['sum_click'], 
                                                         bins=3, 
                                                         labels=['Low', 'Medium', 'High'],
                                                         include_lowest=True,
                                                         duplicates='drop')
                        df['participation_level'] = df['participation_level'].fillna('Low')
                        df['participation_level'] = preprocessing.LabelEncoder().fit_transform(df['participation_level'].astype(str))
                        interaction_features += 1
                        print(f"   - participation_level creada")
                    
                    # Log transformation para reducir skewness (versión segura)
                    # Usar transformación manual más segura
                    df['log_sum_click'] = df['sum_click'].apply(lambda x: np.log(x + 1) if pd.notnull(x) and x >= 0 else 0)
                    interaction_features += 1
                    print(f"   - log_sum_click creada")
                    
                except Exception as e:
                    print(f"   - Error creando características de sum_click: {e}")
            
            # Características socioeconómicas combinadas
            if 'imd_band' in df.columns and 'highest_education' in df.columns:
                try:
                    imd_numeric = pd.to_numeric(df['imd_band'], errors='coerce').fillna(0)
                    edu_numeric = pd.to_numeric(df['highest_education'], errors='coerce').fillna(0)
                    df['socioeconomic_education_score'] = imd_numeric * 0.6 + edu_numeric * 0.4
                    interaction_features += 1
                    print(f"   - socioeconomic_education_score creada")
                except Exception as e:
                    print(f"   - Error creando socioeconomic_education_score: {e}")
            
            # Características de experiencia académica
            if 'num_of_prev_attempts' in df.columns:
                try:
                    attempts = pd.to_numeric(df['num_of_prev_attempts'], errors='coerce').fillna(0)
                    df['is_first_attempt'] = (attempts == 0).astype(int)
                    df['is_experienced_student'] = (attempts >= 2).astype(int)
                    interaction_features += 2
                    print(f"   - Características de experiencia creadas")
                except Exception as e:
                    print(f"   - Error creando características de experiencia: {e}")
        
        print(f"   - Características de interacción creadas: {interaction_features}")
        
        # 4. CARACTERÍSTICAS ESTADÍSTICAS POR GRUPO (simplificado)
        if advanced_features and len(df) > 1000:  # Solo para datasets grandes
            print("4. Creando características estadísticas por grupo...")
            
            group_features = 0
            
            try:
                # Estadísticas por módulo (versión simplificada)
                if 'code_module' in df.columns and 'sum_click' in df.columns:
                    # Asegurar tipos numéricos
                    df['sum_click'] = pd.to_numeric(df['sum_click'], errors='coerce').fillna(0)
                    
                    module_stats = df.groupby('code_module')['sum_click'].agg(['mean']).reset_index()
                    module_stats.columns = ['code_module', 'module_avg_clicks']
                    df = df.merge(module_stats, on='code_module', how='left')
                    
                    # Comparación con promedio del módulo
                    df['clicks_vs_module_avg'] = df['sum_click'] - df['module_avg_clicks']
                    group_features += 2
                    print(f"   - Características de módulo creadas")
                
            except Exception as e:
                print(f"   - Error creando características de grupo: {e}")
            
            print(f"   - Características de grupo creadas: {group_features}")
        
        # 5. ESCALADO OPCIONAL DE CARACTERÍSTICAS NUMÉRICAS
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['final_result', 'id_student']]
        
        if len(numeric_cols) > 0:
            print(f"5. Información de características numéricas: {len(numeric_cols)} variables")
            
            # Guardar estadísticas para referencia
            try:
                self.feature_stats = {
                    'numeric_features': list(numeric_cols),
                    'feature_ranges': {col: {
                        'min': float(df[col].min()), 
                        'max': float(df[col].max()), 
                        'mean': float(df[col].mean())
                    } for col in numeric_cols if df[col].dtype in ['int64', 'float64', 'int32', 'float32']}
                }
            except Exception as e:
                print(f"   - Error calculando estadísticas: {e}")
        
        # 6. REPORTE FINAL
        final_features = df.shape[1]
        features_added = final_features - initial_features
        
        print(f"\nRESUMEN DE FEATURE ENGINEERING:")
        print(f"   - Características iniciales: {initial_features}")
        print(f"   - Características finales: {final_features}")
        print(f"   - Características añadidas: {features_added}")
        print(f"   - Variables numéricas: {len(numeric_cols)}")
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Devuelve el reporte de limpieza de datos."""
        return getattr(self, 'cleaning_report', {})
    
    def get_feature_stats(self) -> Dict:
        """Devuelve las estadísticas de características creadas."""
        return getattr(self, 'feature_stats', {})
    
    def save_preprocessing_report(self, output_path: str):
        """Guarda un reporte completo del preprocesamiento."""
        try:
            report = {
                'cleaning_report': self.get_cleaning_report(),
                'feature_stats': self.get_feature_stats(),
                'outlier_bounds': self.outlier_bounds,
                'encoders_info': {col: list(encoder.classes_) for col, encoder in self.encoders.items()}
            }
            
            # Convertir a DataFrame para guardar
            report_df = pd.DataFrame([{
                'metric': 'initial_records',
                'value': report['cleaning_report'].get('initial_records', 0)
            }, {
                'metric': 'final_records', 
                'value': report['cleaning_report'].get('final_records', 0)
            }, {
                'metric': 'data_quality_score',
                'value': report['cleaning_report'].get('data_quality_score', 0)
            }, {
                'metric': 'features_added',
                'value': len(report['feature_stats'].get('numeric_features', []))
            }])
            
            report_df.to_csv(f"{output_path}/preprocessing_report.csv", index=False)
            print(f"Reporte de preprocesamiento guardado en: {output_path}/preprocessing_report.csv")
        except Exception as e:
            print(f"Error guardando reporte: {e}")