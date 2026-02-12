import cv2
import numpy as np
from matplotlib import pyplot as plt

# ============================================================================
# CONFIGURACIÓN: Valores de entrada/salida y efecto a aplicar
# ============================================================================

IMAGEN_ENTRADA = "photo.jpg"            # Foto con personas
IMAGEN_SALIDA = "output-photo.jpg"      # Resultado

# Elige qué efecto aplicar (descomenta solo UNO):
# EFECTO = "rectangulo"    # Dibuja rectángulos verdes en caras
# EFECTO = "pixelar"     # Pixela las caras (privacidad)
# EFECTO = "blur"        # Difumina las caras
EFECTO = "resaltar"    # Resalta caras en color, resto gris

# ============================================================================
# FUNCIONES
# ============================================================================

def cargar_detector():
    """Carga el detector de rostros (viene con OpenCV)"""
    print("Cargando detector de rostros...")
    
    # Este archivo viene incluido con OpenCV
    ruta_detector = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(ruta_detector)
    
    if detector.empty():
        print("❌ Error: No se pudo cargar el detector")
        return None
    
    print("✓ Detector cargado")
    return detector


def leer_imagen(ruta):
    """Lee la imagen"""
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"❌ Error: No se encontró '{ruta}'")
        return None
    print(f"✓ Imagen cargada: {imagen.shape}")
    return imagen


def detectar_rostros(imagen, detector):
    """Detecta todos los rostros en la imagen"""
    print("Buscando rostros...")
    
    # Convertir a escala de grises (el detector funciona mejor así)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    # Retorna: lista de (x, y, ancho, alto) de cada rostro
    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,    # Qué tan rápido buscar
        minNeighbors=5,      # Cuántas detecciones para confirmar
        minSize=(30, 30)     # Tamaño mínimo del rostro
    )
    
    print(f"✓ Encontrados {len(rostros)} rostro(s)")
    return rostros


def dibujar_rectangulos(imagen, rostros):
    """Dibuja un rectángulo verde alrededor de cada rostro"""
    print("Dibujando rectángulos en rostros...")
    
    resultado = imagen.copy()
    
    for i, (x, y, ancho, alto) in enumerate(rostros, 1):
        # Dibujar rectángulo verde
        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)
        
        # Agregar número del rostro
        cv2.putText(resultado, f"Rostro {i}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return resultado


def pixelar_rostros(imagen, rostros, nivel=15):
    """Pixela los rostros para proteger privacidad"""
    print("Pixelando rostros...")
    
    resultado = imagen.copy()
    
    for (x, y, ancho, alto) in rostros:
        # Extraer la región del rostro
        rostro = resultado[y:y+alto, x:x+ancho]
        
        # Reducir tamaño (hace píxeles grandes)
        pequeño = cv2.resize(rostro, (nivel, nivel), interpolation=cv2.INTER_LINEAR)
        
        # Volver a agrandar (efecto pixelado)
        pixelado = cv2.resize(pequeño, (ancho, alto), interpolation=cv2.INTER_NEAREST)
        
        # Reemplazar en la imagen
        resultado[y:y+alto, x:x+ancho] = pixelado
    
    return resultado


def desenfocar_rostros(imagen, rostros):
    """Difumina los rostros"""
    print("Difuminando rostros...")
    
    resultado = imagen.copy()
    
    for (x, y, ancho, alto) in rostros:
        # Extraer rostro
        rostro = resultado[y:y+alto, x:x+ancho]
        
        # Aplicar blur fuerte
        difuminado = cv2.GaussianBlur(rostro, (99, 99), 30)
        
        # Reemplazar en la imagen
        resultado[y:y+alto, x:x+ancho] = difuminado
    
    return resultado


def resaltar_rostros(imagen, rostros):
    """Resalta rostros en color, resto en blanco y negro"""
    print("Resaltando rostros...")
    
    # Convertir imagen completa a blanco y negro
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    resultado = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    
    # Poner rostros originales en color
    for (x, y, ancho, alto) in rostros:
        resultado[y:y+alto, x:x+ancho] = imagen[y:y+alto, x:x+ancho]
    
    return resultado


def guardar_imagen(imagen, ruta):
    """Guarda la imagen"""
    cv2.imwrite(ruta, imagen)
    print(f"✓ Imagen guardada: {ruta}")


def mostrar_comparacion(original, procesada, num_rostros):
    """Muestra antes y después"""
    print("Generando comparación...")
    
    # Convertir a RGB para matplotlib
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(orig_rgb)
    plt.title('IMAGEN ORIGINAL', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(proc_rgb)
    plt.title(f'ROSTROS DETECTADOS: {num_rostros}', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparacion_rostros.png', dpi=150, bbox_inches='tight')
    print("✓ Comparación guardada: comparacion_rostros.png")
    plt.show()


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*70)
    print("            DETECTOR DE ROSTROS - VERSIÓN SIMPLE")
    print("="*70 + "\n")
    
    # PASO 1: Cargar el detector
    detector = cargar_detector()
    if detector is None:
        return
    
    # PASO 2: Cargar imagen
    imagen = leer_imagen(IMAGEN_ENTRADA)
    if imagen is None:
        print("\n💡 CONSEJO: Coloca una foto con personas y renómbrala a 'foto.jpg'")
        print("   O cambia IMAGEN_ENTRADA en la línea 14\n")
        return
    
    # PASO 3: Detectar rostros
    print("\n" + "-"*70)
    rostros = detectar_rostros(imagen, detector)
    print("-"*70 + "\n")
    
    if len(rostros) == 0:
        print("⚠️  No se detectaron rostros en la imagen")
        print("\n💡 CONSEJOS:")
        print("   • Usa fotos donde las caras sean claras y frontales")
        print("   • Asegúrate de que haya buena iluminación")
        print("   • Las caras no deben estar muy de perfil\n")
        return
    
    # PASO 4: Aplicar el efecto elegido
    print(f"Aplicando efecto: {EFECTO}\n")
    
    if EFECTO == "rectangulo":
        resultado = dibujar_rectangulos(imagen, rostros)
    elif EFECTO == "pixelar":
        resultado = pixelar_rostros(imagen, rostros, nivel=15)
    elif EFECTO == "blur":
        resultado = desenfocar_rostros(imagen, rostros)
    elif EFECTO == "resaltar":
        resultado = resaltar_rostros(imagen, rostros)
    else:
        print(f"❌ Efecto '{EFECTO}' no reconocido")
        print("   Usa: rectangulo, pixelar, blur o resaltar")
        return
    
    # PASO 5: Guardar y mostrar
    guardar_imagen(resultado, IMAGEN_SALIDA)
    print()
    mostrar_comparacion(imagen, resultado, len(rostros))
    
    print("\n" + "="*70)
    print("✓ PROCESO COMPLETADO")
    print("="*70)
    print(f"\nRostros detectados: {len(rostros)}")
    print(f"Efecto aplicado: {EFECTO}")
    print(f"\nArchivos generados:")
    print(f"  • {IMAGEN_SALIDA}")
    print(f"  • comparacion_rostros.png")
    print()


if __name__ == "__main__":
    main()