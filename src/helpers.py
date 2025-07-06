# Variables de color y función auxiliar para impresión de secciones
color_azul_destacado = "\033[94m\033[1m"
color_rojo_alerta = "\033[91m"
color_reset = "\033[0m"


# Funciones auxiliares reutilizables

# Función auxiliar para mostrar secciones

def display_section_header(title: str):
    print(f"\n{'='*10} {title} {'='*10}\n")
