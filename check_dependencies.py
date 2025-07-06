#!/usr/bin/env python3
"""
check_dependencies.py
Script para verificar que todas las dependencias estén instaladas correctamente.
"""

import sys
import importlib

required_packages = [
    'pandas',
    'numpy', 
    'sklearn',
    'matplotlib',
    'seaborn',
    'mysql.connector',
    'tqdm',
    'imblearn',
    'sqlalchemy',
    'xgboost',
    'scipy',
    'joblib',
    'warnings'
]

def check_package(package_name):
    """Verifica si un paquete está instalado."""
    try:
        if package_name == 'sklearn':
            importlib.import_module('sklearn')
        elif package_name == 'mysql.connector':
            importlib.import_module('mysql.connector')
        else:
            importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    """Función principal para verificar dependencias."""
    print("Verificando dependencias del proyecto...")
    print("=" * 50)
    
    missing_packages = []
    
    for package in required_packages:
        if check_package(package):
            print(f"✓ {package}")
        else:
            print(f"✗ {package} - NO ENCONTRADO")
            missing_packages.append(package)
    
    print("=" * 50)
    
    if missing_packages:
        print(f"ADVERTENCIA: {len(missing_packages)} paquetes faltantes:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstalar con:")
        print("pip install " + " ".join(missing_packages))
        return False
    else:
        print("✓ Todas las dependencias están instaladas correctamente")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
