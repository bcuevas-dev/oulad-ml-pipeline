"""
query_executor.py
Funciones para consultas y lecturas SQL usando DatabaseManager
"""
import pandas as pd
from connection_manager import DatabaseManager

# Función para leer una tabla completa a DataFrame
def fetch_table_data(table_name: str, db_manager: DatabaseManager) -> pd.DataFrame:
    engine = db_manager.get_engine()
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(f"Leídos {len(df)} registros de la tabla {table_name}.")
    return df

# Función para ejecutar una consulta SQL personalizada
def execute_custom_query(query: str, db_manager: DatabaseManager) -> pd.DataFrame:
    engine = db_manager.get_engine()
    df = pd.read_sql(query, engine)
    print(f"Consulta ejecutada. Registros obtenidos: {len(df)}")
    return df

# Función para obtener solo los nombres de las tablas
def get_table_list(db_manager: DatabaseManager) -> list:
    from sqlalchemy import inspect
    engine = db_manager.get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tablas en la base de datos: {tables}")
    return tables
