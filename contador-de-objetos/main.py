import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACIÓN - Valores iniciales
# ============================================================================

IMAGEN_ENTRADA = "photo-1.jpg"
IMAGEN_SALIDA = "objetos_contados.jpg"

# PARÁMETROS BALANCEADOS (ajusta para tu imagen)
UMBRAL_BORDES = 30           # Sensibilidad de detección (20-50)
ITERACIONES_CIERRE = 3       # ← REDUCIDO: Une fragmentos sin unir frutas (2-4)
AREA_MINIMA = 300            # Tamaño mínimo de objeto (200-800)
SEPARAR_OBJETOS_GRANDES = True  # Intenta separar frutas pegadas

# ============================================================================
# FUNCIONES
# ============================================================================

def leer_imagen(ruta):
    """Lee imagen"""
    print(f"📂 Cargando: {ruta}")
    try:
        img = Image.open(ruta)
        img_gris = img.convert('L')
        imagen = np.array(img_gris, dtype=np.float64)
        print(f"✓ Cargada: {imagen.shape}")
        return imagen, img
    except:
        print(f"❌ No encontrada: '{ruta}'")
        return None, None


def crear_kernel_gaussiano(tamaño=5, sigma=1.0):
    """Kernel gaussiano"""
    kernel = np.zeros((tamaño, tamaño))
    centro = tamaño // 2
    for i in range(tamaño):
        for j in range(tamaño):
            x, y = i - centro, j - centro
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def aplicar_convolucion(imagen, kernel):
    """Convolución manual"""
    filas, columnas = imagen.shape
    k_filas, k_columnas = kernel.shape
    pad = k_filas // 2
    imagen_pad = np.pad(imagen, pad, mode='edge')
    resultado = np.zeros_like(imagen)
    
    for i in range(filas):
        for j in range(columnas):
            region = imagen_pad[i:i+k_filas, j:j+k_columnas]
            resultado[i, j] = np.sum(region * kernel)
    
    return resultado


def detectar_bordes_sobel(imagen):
    """Sobel manual"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = aplicar_convolucion(imagen, sobel_x)
    gy = aplicar_convolucion(imagen, sobel_y)
    
    return np.sqrt(gx**2 + gy**2)


def detectar_bordes(imagen, umbral=30):
    """Detecta bordes"""
    print("🔍 Detectando bordes...")
    kernel = crear_kernel_gaussiano(5, 1.5)
    suave = aplicar_convolucion(imagen, kernel)
    magnitud = detectar_bordes_sobel(suave)
    bordes = (magnitud > umbral).astype(np.uint8) * 255
    return bordes


def cerrar_bordes_moderado(imagen, iteraciones=3):
    """
    Cierre morfológico MODERADO
    CLAVE: Solo 2-4 iteraciones para NO unir frutas cercanas
    """
    print(f"🔗 Cerrando bordes (moderado, {iteraciones} iter)...")
    
    resultado = imagen.copy()
    kernel_size = 3  # Kernel pequeño
    
    # DILATACIÓN
    for _ in range(iteraciones):
        temp = np.zeros_like(resultado)
        filas, columnas = resultado.shape
        
        for i in range(1, filas-1):
            for j in range(1, columnas-1):
                region = resultado[i-1:i+2, j-1:j+2]
                if np.any(region > 0):
                    temp[i, j] = 255
        resultado = temp
    
    # EROSIÓN
    for _ in range(iteraciones):
        temp = np.zeros_like(resultado)
        filas, columnas = resultado.shape
        
        for i in range(1, filas-1):
            for j in range(1, columnas-1):
                region = resultado[i-1:i+2, j-1:j+2]
                if np.sum(region > 0) >= 5:
                    temp[i, j] = 255
        resultado = temp
    
    return resultado


def analizar_forma(mascara, bbox):
    """
    Analiza si un objeto tiene forma irregular (probablemente sean 2+ frutas)
    Retorna True si parece ser múltiples objetos
    """
    x1, y1, x2, y2 = bbox
    ancho = x2 - x1
    alto = y2 - y1
    
    if ancho <= 0 or alto <= 0:
        return False
    
    # Ratio de aspecto
    ratio = max(ancho, alto) / min(ancho, alto)
    
    # Área del bounding box vs área real
    area_bbox = ancho * alto
    area_real = np.sum(mascara > 0)
    
    if area_bbox == 0:
        return False
    
    solidez = area_real / area_bbox
    
    # Si es muy alargado Y poco sólido → probablemente 2+ objetos
    if ratio > 2.5 and solidez < 0.6:
        return True
    
    # Si es MUY grande comparado con otros
    if area_real > 10000:  # Muy grande
        return True
    
    return False


def intentar_separar_objeto(imagen, etiqueta, mascara, bbox):
    """
    Intenta separar un objeto grande en partes usando erosión
    """
    x1, y1, x2, y2 = bbox
    
    # Extraer región
    region = mascara[y1:y2+1, x1:x2+1].copy()
    
    # Aplicar erosión para separar
    for _ in range(3):
        temp = np.zeros_like(region)
        h, w = region.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if region[i, j] > 0:
                    vecinos = region[i-1:i+2, j-1:j+2]
                    if np.sum(vecinos > 0) >= 7:  # Mantener solo centros
                        temp[i, j] = 255
        
        region = temp
    
    # Etiquetar componentes en la región erosionada
    sub_objetos = []
    visitado = np.zeros_like(region, dtype=bool)
    h, w = region.shape
    
    for i in range(h):
        for j in range(w):
            if region[i, j] > 0 and not visitado[i, j]:
                # Flood fill para encontrar componente
                stack = [(i, j)]
                puntos = []
                
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= h or cj < 0 or cj >= w:
                        continue
                    if visitado[ci, cj] or region[ci, cj] == 0:
                        continue
                    
                    visitado[ci, cj] = True
                    puntos.append((ci + y1, cj + x1))
                    
                    stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])
                
                if len(puntos) > 50:  # Solo si es suficientemente grande
                    sub_objetos.append(puntos)
    
    return sub_objetos if len(sub_objetos) > 1 else None


def etiquetar_objetos_inteligente(imagen, min_area=300, separar=True):
    """
    Etiqueta objetos E INTENTA SEPARAR los que parecen múltiples
    """
    print(f"🏷️  Etiquetando objetos (área mín: {min_area})...")
    
    filas, columnas = imagen.shape
    etiquetas = np.zeros((filas, columnas), dtype=np.int32)
    etiqueta_actual = 1
    objetos = []
    
    for i in range(filas):
        for j in range(columnas):
            if imagen[i, j] > 0 and etiquetas[i, j] == 0:
                # Etiquetar objeto
                area, bbox = etiquetar_uno(imagen, etiquetas, i, j, etiqueta_actual)
                
                if area >= min_area:
                    # Extraer máscara del objeto
                    mascara = (etiquetas == etiqueta_actual).astype(np.uint8) * 255
                    
                    # ¿Es sospechoso? (múltiples frutas juntas)
                    if separar and analizar_forma(mascara, bbox):
                        print(f"   ⚠️  Objeto {etiqueta_actual} parece múltiple, intentando separar...")
                        
                        sub_objetos = intentar_separar_objeto(imagen, etiqueta_actual, mascara, bbox)
                        
                        if sub_objetos:
                            print(f"      → Separado en {len(sub_objetos)} partes")
                            
                            # Borrar etiqueta original
                            etiquetas[etiquetas == etiqueta_actual] = 0
                            
                            # Crear objetos separados
                            for puntos in sub_objetos:
                                # Calcular bbox de cada sub-objeto
                                min_x = min(p[1] for p in puntos)
                                max_x = max(p[1] for p in puntos)
                                min_y = min(p[0] for p in puntos)
                                max_y = max(p[0] for p in puntos)
                                
                                # Re-etiquetar en la imagen original
                                for pi, pj in puntos:
                                    etiquetas[pi, pj] = etiqueta_actual
                                
                                objetos.append({
                                    'etiqueta': etiqueta_actual,
                                    'area': len(puntos),
                                    'bbox': (min_x, min_y, max_x, max_y),
                                    'centro': ((min_x + max_x) // 2, (min_y + max_y) // 2)
                                })
                                
                                etiqueta_actual += 1
                        else:
                            # No se pudo separar, mantener como está
                            objetos.append({
                                'etiqueta': etiqueta_actual,
                                'area': area,
                                'bbox': bbox,
                                'centro': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                            })
                            etiqueta_actual += 1
                    else:
                        # Objeto normal
                        objetos.append({
                            'etiqueta': etiqueta_actual,
                            'area': area,
                            'bbox': bbox,
                            'centro': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        })
                        etiqueta_actual += 1
                else:
                    # Borrar ruido
                    etiquetas[etiquetas == etiqueta_actual] = 0
    
    print(f"✓ {len(objetos)} objetos detectados")
    return etiquetas, objetos


def etiquetar_uno(imagen, etiquetas, start_i, start_j, etiqueta):
    """Etiqueta un objeto"""
    filas, columnas = imagen.shape
    stack = [(start_i, start_j)]
    area = 0
    min_x, min_y = columnas, filas
    max_x, max_y = 0, 0
    
    while stack:
        i, j = stack.pop()
        
        if i < 0 or i >= filas or j < 0 or j >= columnas:
            continue
        if etiquetas[i, j] != 0 or imagen[i, j] == 0:
            continue
        
        etiquetas[i, j] = etiqueta
        area += 1
        
        min_x, max_x = min(min_x, j), max(max_x, j)
        min_y, max_y = min(min_y, i), max(max_y, i)
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di != 0 or dj != 0:
                    stack.append((i + di, j + dj))
    
    return area, (min_x, min_y, max_x, max_y)


def dibujar_resultados(imagen_original, objetos):
    """Dibuja resultados"""
    print("✏️  Dibujando...")
    
    if isinstance(imagen_original, np.ndarray):
        img = Image.fromarray(imagen_original.astype(np.uint8)).convert('RGB')
    else:
        img = imagen_original.convert('RGB')
    
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    colores = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
    ]
    
    for idx, obj in enumerate(objetos):
        x1, y1, x2, y2 = obj['bbox']
        color = colores[idx % len(colores)]
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        numero = str(idx + 1)
        bbox_text = draw.textbbox((0, 0), numero, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        
        cx, cy = obj['centro']
        tx, ty = cx - tw // 2, cy - th // 2
        
        draw.rectangle([tx-5, ty-5, tx+tw+5, ty+th+5], fill=(255, 255, 255))
        draw.text((tx, ty), numero, fill=color, font=font)
    
    return img


def guardar(imagen, ruta):
    """Guarda imagen"""
    if isinstance(imagen, np.ndarray):
        Image.fromarray(imagen.astype(np.uint8)).save(ruta)
    else:
        imagen.save(ruta)
    print(f"💾 {ruta}")


def mostrar_proceso(original, bordes, cerrados, final, num):
    """Visualización"""
    print("📊 Generando visualización...")
    
    fig = plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('1. ORIGINAL', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(bordes, cmap='gray')
    plt.title('2. BORDES', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cerrados, cmap='gray')
    plt.title(f'3. CERRADOS\n({ITERACIONES_CIERRE} iter)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(final)
    plt.title(f'5. RESULTADO\n✓ {num} OBJETOS', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.axis('off')
    info = f"""PARÁMETROS:
• Umbral: {UMBRAL_BORDES}
• Cierre: {ITERACIONES_CIERRE} iter
• Área mín: {AREA_MINIMA} px
• Separación: {"Sí" if SEPARAR_OBJETOS_GRANDES else "No"}

DETECTADOS: {num}

AJUSTAR:
- Muy pocos → Bajar área_mínima
- Muy juntos → Bajar iteraciones
- Muy separados → Subir iteraciones
    """
    plt.text(0.1, 0.5, info, fontsize=11, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig('proceso_final.png', dpi=150, bbox_inches='tight')
    print("💾 proceso_final.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("     CONTADOR BALANCEADO - Une fragmentos Y separa cercanos")
    print("="*70 + "\n")
    
    imagen_gris, imagen_color = leer_imagen(IMAGEN_ENTRADA)
    if imagen_gris is None:
        return
    
    print(f"\n⚙️  PARÁMETROS:")
    print(f"   Umbral: {UMBRAL_BORDES} | Cierre: {ITERACIONES_CIERRE} | Área: {AREA_MINIMA}\n")
    
    # Procesar
    bordes = detectar_bordes(imagen_gris, UMBRAL_BORDES)
    cerrados = cerrar_bordes_moderado(bordes, ITERACIONES_CIERRE)
    etiquetas, objetos = etiquetar_objetos_inteligente(cerrados, AREA_MINIMA, SEPARAR_OBJETOS_GRANDES)
    final = dibujar_resultados(imagen_color, objetos)
    
    # Guardar
    print()
    guardar(bordes, "1_bordes.jpg")
    guardar(cerrados, "2_cerrados.jpg")
    guardar(final, IMAGEN_SALIDA)
    
    # Mostrar
    print()
    mostrar_proceso(imagen_gris, bordes, cerrados, final, len(objetos))
    
    print("\n" + "="*70)
    print(f"✓ TOTAL: {len(objetos)} OBJETOS")
    print("="*70 + "\n")
    
    for idx, obj in enumerate(objetos, 1):
        x1, y1, x2, y2 = obj['bbox']
        print(f"   #{idx}: {obj['area']:5d} px | {x2-x1}x{y2-y1} px")
    
    print("\n📁 Archivos:")
    print("   • objetos_contados.jpg\n   • proceso_final.png\n")


if __name__ == "__main__":
    main()