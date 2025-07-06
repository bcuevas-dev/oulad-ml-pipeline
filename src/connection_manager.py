"""
connection_manager.py
Clase para gestionar la conexión a la base de datos usando SQLAlchemy y database_config.py
"""
import pandas as pd
from sqlalchemy import create_engine
import sys
import importlib.util
import os

class DatabaseManager:
    """
    Clase para gestionar la conexión a la base de datos usando SQLAlchemy.
    Permite cargar la configuración desde un archivo database_config.py externo.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Ruta por defecto relativa al proyecto
            config_path = os.path.join(os.path.dirname(__file__), '../config/settings.py')
        spec = importlib.util.spec_from_file_location("oulad_settings", config_path)
        settings = importlib.util.module_from_spec(spec)
        sys.modules["oulad_settings"] = settings
        spec.loader.exec_module(settings)
        self.settings = settings
        try:
            self.engine = create_engine(settings.SQLALCHEMY_URL)
        except Exception as e:
            print(f"Error creando el engine de SQLAlchemy: {e}")
            self.engine = None

    def get_engine(self):
        """Devuelve el engine de SQLAlchemy."""
        return self.engine

    def test_connection(self):
        """Prueba la conexión a la base de datos y muestra un mensaje de éxito o error."""
        if self.engine is None:
            print("Engine no inicializado.")
            return
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            print("¡Conexión exitosa!")
        except Exception as e:
            print(f"Error de conexión: {e}")

    def export_to_database(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace', index: bool = False):
        """Exporta un DataFrame a una tabla SQL."""
        if self.engine is None:
            print("Engine no inicializado. No se puede exportar a SQL.")
            return
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            print(f"Datos exportados a la tabla {table_name}.")
        except Exception as e:
            print(f"Error exportando datos a SQL: {e}")

if __name__ == "__main__":
    print("Probando conexión a la base de datos...")
    db = DatabaseManager()
    db.test_connection()
