import cv2
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1. CATEGORIZACIÓN Y CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
PESOS = {
    # Transitable
    7: 0.0, 8: 0.0, 
    
    # Intransitable
    1: 0.8, 2: 0.1, 3: 0.4, 4: 0.25, 
    5: 1.0, 6: 1.0, 9: 0.7, 10: 0.7,

    # Incertidumbre
    0: 0.5
}

EXTENSIONES_IMAGEN = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def crear_carpeta_ejecucion(base_salidas: Path) -> Path:
    base_salidas.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_ejecucion = base_salidas / f"mapa_riesgo_{timestamp}"
    carpeta_ejecucion.mkdir(parents=True, exist_ok=True)
    return carpeta_ejecucion

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERACIÓN DE KERNEL CÓNICO
# ─────────────────────────────────────────────────────────────────────────────
def obtener_kernel_conico(radio):
    """Crea una matriz cuadrada donde los valores decaen linealmente desde el centro"""
    y, x = np.ogrid[-radio:radio+1, -radio:radio+1]
    distancia = np.sqrt(x**2 + y**2)
    kernel = np.maximum(0, 1 - (distancia / radio))
    return kernel

# ─────────────────────────────────────────────────────────────────────────────
# 3. NÚCLEO: MAPA DE COSTES POR CONVOLUCIÓN
# ─────────────────────────────────────────────────────────────────────────────
def generar_mapa_bicolor(ruta_imagen: Path, model: YOLO, carpeta_salida: Path, radio_max: int = 150, mostrar: bool = False):
    resultados = model.predict(str(ruta_imagen), show=False, verbose=False, imgsz=640, conf=0.15)[0]
    img_original = resultados.orig_img.copy()
    h, w = img_original.shape[:2]
    mapa_pesos_nitido = np.full((h, w), PESOS[0], dtype=np.float32)
    mascara_transitable = np.zeros((h, w), dtype=bool)
    mascara_vehiculos = np.zeros((h, w), dtype=np.uint8)
    mascara_para_inflar = np.zeros((h, w), dtype=np.float32)

    if resultados.masks is not None:
        mascaras = resultados.masks.data.cpu().numpy()
        clases = resultados.boxes.cls.cpu().numpy()
        for i, cls_id in enumerate(clases):
            id_clase = int(cls_id)
            if id_clase == 0: continue
            
            mask_resized = cv2.resize(mascaras[i], (w, h))
            mask_bool = mask_resized > 0.5
            
            if id_clase == 6:  
                radio_exp = 1 
                d = 2 * radio_exp + 1
                kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
                mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel_circular, iterations=1).astype(bool)

            if id_clase == 6:
                mascara_vehiculos[mask_bool] = 255

            peso_asignado = PESOS.get(id_clase, 0.5)
            mapa_pesos_nitido[mask_bool] = peso_asignado
            if id_clase in [5, 6, 9]:
                mascara_para_inflar[mask_bool] = 1.0
            if peso_asignado < 0.4: 
                mascara_transitable[mask_bool] = True

    # --- INFLADO CÓNICO INTELIGENTE ---
    if cv2.countNonZero(mascara_para_inflar) > 0:
        kernel = obtener_kernel_conico(radio_max)
        mapa_inflado = cv2.filter2D(mascara_para_inflar, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        mapa_final = np.maximum(mapa_pesos_nitido, mapa_inflado)
    else:
        mapa_final = mapa_pesos_nitido

    # --- RENDERIZADO VISUAL ---
    mapa_final = np.clip(mapa_final, 0.0, 1.0)
    h, w = mapa_final.shape
    mapa_color_bgr = np.ones((h, w, 3), dtype=np.uint8) * 255
    factor_color = (1.0 - mapa_final) * 255
    mapa_color_bgr[:, :, 0] = factor_color.astype(np.uint8)
    mapa_color_bgr[:, :, 1] = factor_color.astype(np.uint8)
    if mostrar:
        contornos, _ = cv2.findContours(mascara_vehiculos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contorno in contornos:
            for j in range(0, len(contorno), 15):
                if j + 8 < len(contorno):
                    pt1 = tuple(contorno[j][0])
                    pt2 = tuple(contorno[j+8][0])
                    cv2.line(img_original, pt1, pt2, (0, 255, 255), 6)
                    cv2.line(mapa_color_bgr, pt1, pt2, (0, 255, 255), 6)

    comparacion = np.hstack((img_original, mapa_color_bgr))
    
    ruta_salida = carpeta_salida / f"mapchart_{ruta_imagen.name}"
    cv2.imwrite(str(ruta_salida), comparacion)
    return ruta_salida
# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RV-MAP: Kernel Cónico")
    parser.add_argument("--input", "-i", type=str, default="input_dron", help="Carpeta de entrada.")
    parser.add_argument("--output", "-o", type=str, default="output_mapas", help="Carpeta de salida.")
    parser.add_argument("--model", "-m", type=str, default="modelos/model_C/weights/best.pt", help="Modelo YOLO.")
    parser.add_argument("--radio", "-r", type=int, default=100, help="Radio del cono en px.")
    parser.add_argument("--show", action="store_true", default=False, help="Mostrar el área de interés (vehículos).")
    args = parser.parse_args()

    carpeta_input = Path(args.input)
    if not carpeta_input.exists():
        print(f"Error: No existe la carpeta {args.input}")
        return

    imagenes = sorted([f for f in carpeta_input.iterdir() if f.suffix.lower() in EXTENSIONES_IMAGEN])
    if not imagenes:
        print("No se encontraron imágenes.")
        return

    carpeta_salida = crear_carpeta_ejecucion(Path(args.output))
    model = YOLO(args.model)

    for ruta_img in tqdm(imagenes, desc="Procesando imágenes", unit="img"):
        try:
            generar_mapa_bicolor(ruta_img, model, carpeta_salida, radio_max=args.radio, mostrar=args.show)
        except Exception as e:
            print(f"\nError en {ruta_img.name}: {e}")

if __name__ == "__main__":
    main()
