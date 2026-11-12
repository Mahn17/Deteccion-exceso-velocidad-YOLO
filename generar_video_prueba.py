#Manuel Alejandro Heredia Nogales
"""Genera un video de prueba con velocidades conocidas

Usa la foto de la camioneta de la figura recursos/non-max-suppression.png, que se extrajo
de py/yolo/v1_autonomous_driving_application_car_detection.py (curso Deep Learning de
Coursera; foto del dataset de drive.ai):
  1. Recorta la foto de la calle y borra la camioneta del fondo (inpainting).
  2. Pega dos copias de la camioneta que cruzan la calle a velocidad constante,
     cada una con una placa dibujada.
  3. Calcula el movimiento para que entre la línea A (30 % del ancho) y la línea B
     (70 %) haya exactamente 20 m.

Resultado esperado con la configuración que se imprime al final:
  VSA123B a 90 km/h -> multa;  WKB4521 a 45 km/h -> sin multa.

Uso:
    python generar_video_prueba.py              # crea video_prueba.mp4
"""

# =============================================================================
# IMPORTACIONES

import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# CONFIGURACIÓN

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(DIR_BASE, 'recursos', 'non-max-suppression.png')
SALIDA = os.path.join(DIR_BASE, 'video_prueba.mp4')

ANCHO, ALTO, FPS, DURACION = 1280, 720, 30, 6.5
LINEA_A, LINEA_B, DISTANCIA_M = 0.3, 0.7, 20.0
PX_POR_M = (LINEA_B - LINEA_A) * ANCHO / DISTANCIA_M      # 25.6 píxeles por metro

# (placa, velocidad km/h, dirección +1 = hacia la derecha, escala, y de las llantas, segundo en que entra)
AUTOS = [
    ('VSA123B', 90, +1, 1.0, 700, 1.0),
    ('WKB4521', 45, -1, 0.8, 470, 0.0),
]

FUENTES = ['/usr/share/fonts/liberation/LiberationSans-Bold.ttf',
           '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
           'DejaVuSans-Bold.ttf']


# =============================================================================
# FUNCIONES

def foto_del_repositorio():
    """Lee la figura con la foto de la calle (dos paneles: antes y después de non-max suppression)."""
    figura = cv2.imread(ORIGEN, cv2.IMREAD_COLOR)
    if figura is None:
        raise SystemExit(f"No se encontró la figura {ORIGEN}")
    return figura


def separar_calle_y_camioneta(figura):
    """Regresa (fondo de la calle sin la camioneta, recorte de la camioneta)."""
    # Panel derecho de la figura ("After non-max suppression"): la foto tiene un marco oscuro
    mitad = figura.shape[1] // 2 + 100
    derecha = cv2.cvtColor(figura[:, mitad:], cv2.COLOR_BGR2GRAY) < 250
    cols = np.where(derecha.mean(axis=0) > 0.5)[0]
    filas = np.where(derecha.mean(axis=1) > 0.5)[0]
    calle = figura[filas[0] + 3:filas[-1] - 2, mitad + cols[0] + 3:mitad + cols[-1] - 2]

    # Recuadro rojo dibujado alrededor de la camioneta (solo cerca de ella, no los semáforos)
    rojo = (calle[:, :, 2] > 150) & (calle[:, :, 1] < 80) & (calle[:, :, 0] < 80)
    zona = np.zeros_like(rojo)
    zona[219:421, 169:423] = True
    ys, xs = np.where(rojo & zona)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    camioneta = calle[y1 + 8:y2 - 7, x1 + 8:x2 - 7].copy()   # sin el recuadro rojo
    mascara = np.zeros(calle.shape[:2], np.uint8)
    mascara[y1 - 4:y2 + 5, x1 - 4:x2 + 5] = 255
    fondo = cv2.inpaint(calle, mascara, 9, cv2.INPAINT_TELEA)
    fondo = cv2.resize(fondo[150:150 + 338, :601], (ANCHO, ALTO), interpolation=cv2.INTER_CUBIC)
    return fondo, camioneta


def fuente(tamano):
    for ruta in FUENTES:
        try:
            return ImageFont.truetype(ruta, tamano)
        except OSError:
            continue
    return ImageFont.load_default(size=tamano)


def con_placa(camioneta, texto):
    """Pinta una placa donde va la placa real de la camioneta."""
    img = camioneta.copy()
    h, w = img.shape[:2]
    pw, ph = int(0.40 * w), int(0.15 * h)
    x1, y1 = int(0.38 * w) - pw // 2, int(0.66 * h) - ph // 2

    placa = Image.new('RGB', (pw, ph), (245, 245, 240))
    d = ImageDraw.Draw(placa)
    f = fuente(int(ph * 0.75))
    bb = d.textbbox((0, 0), texto, font=f)
    d.text(((pw - (bb[2] - bb[0])) / 2 - bb[0], (ph - (bb[3] - bb[1])) / 2 - bb[1]), texto, font=f, fill=(20, 20, 30))
    d.rectangle([0, 0, pw - 1, ph - 1], outline=(40, 40, 40), width=2)
    img[y1:y1 + ph, x1:x1 + pw] = cv2.cvtColor(np.array(placa), cv2.COLOR_RGB2BGR)
    return img


def a_h264(ruta_mp4v, ruta_final):
    """El navegador del celular no reproduce el MPEG-4 de OpenCV; se convierte a H.264 con ffmpeg."""
    if shutil.which('ffmpeg') is None:
        os.replace(ruta_mp4v, ruta_final)
        print("AVISO: no se encontró ffmpeg; el video quedó en MPEG-4 y quizá no se reproduzca en el celular.")
        return
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', ruta_mp4v, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', ruta_final], check=True)
    os.remove(ruta_mp4v)


# =============================================================================
# PROGRAMA PRINCIPAL

if __name__ == '__main__':
    fondo, camioneta = separar_calle_y_camioneta(foto_del_repositorio())
    escala_base = 380 / camioneta.shape[1]      # la camioneta mide 380 píxeles de ancho

    autos = []
    for placa, kmh, direccion, escala, y_llantas, t_entrada in AUTOS:
        sprite = camioneta if direccion > 0 else cv2.flip(camioneta, 1)
        sprite = cv2.resize(sprite, None, fx=escala_base * escala, fy=escala_base * escala,
                            interpolation=cv2.INTER_CUBIC)
        autos.append((con_placa(sprite, placa), kmh, direccion, y_llantas, t_entrada))

    temporal = SALIDA + '.mp4v.mp4'
    out = cv2.VideoWriter(temporal, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (ANCHO, ALTO))
    for n in range(int(DURACION * FPS)):
        t = n / FPS
        cuadro = fondo.copy()
        # Primero el auto más lejano (más arriba), para que el cercano quede encima
        for sprite, kmh, direccion, y_llantas, t_entrada in sorted(autos, key=lambda a: a[3]):
            if t < t_entrada:
                continue
            h, w = sprite.shape[:2]
            recorrido = (t - t_entrada) * kmh / 3.6 * PX_POR_M
            x = int(round(-w + recorrido if direccion > 0 else ANCHO - recorrido))
            y = y_llantas - h
            a, b = max(0, x), min(ANCHO, x + w)
            if a < b:
                cuadro[y:y + h, a:b] = sprite[:, a - x:b - x]
        out.write(cuadro)
    out.release()
    a_h264(temporal, SALIDA)

    print(f"Video: {SALIDA} ({DURACION} s, {ANCHO}x{ALTO}, {FPS} cuadros/s)")
    print("\nConfiguración en la app (⚙ Calibración):")
    print("  ¿Cómo pasan los autos?  De lado (líneas verticales)")
    print(f"  Línea A {LINEA_A:.0%}   Línea B {LINEA_B:.0%}   Distancia {DISTANCIA_M:.0f} m   Límite 60 km/h")
    print("\nResultado esperado:")
    for placa, kmh, *_ in AUTOS:
        print(f"  {placa}: {kmh} km/h  -> {'multa' if kmh > 60 else 'sin multa'}")
