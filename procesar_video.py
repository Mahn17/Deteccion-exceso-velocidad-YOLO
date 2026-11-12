#Manuel Alejandro Heredia Nogales
"""Procesa un video grabado y genera un video con las detecciones y las multas

Equivale a main.py + visualize.py de
py/yolo/Automatic-License-Plate-Recognition-using-YOLOv8-main, en un solo paso.

Uso:
    python procesar_video.py autos.mp4 --distancia 10 --limite 60
    python procesar_video.py autos.mp4 --orientacion horizontal --linea-a 0.55 --linea-b 0.85
"""

# =============================================================================
# IMPORTACIONES

import argparse
import os
import time

import cv2

from deteccion import Configuracion, DetectorExcesoVelocidad, Sesion, dibujar_vehiculo

VERDE, AMARILLO, ROJO, BLANCO = (0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 255, 255)


# =============================================================================
# DIBUJO

def dibujar(frame, resultado, config):
    alto, ancho = frame.shape[:2]

    # Líneas de medición
    for nombre, linea in (('A', config.linea_a), ('B', config.linea_b)):
        if config.orientacion == 'horizontal':
            p1, p2 = (0, int(linea * alto)), (ancho, int(linea * alto))
        else:
            p1, p2 = (int(linea * ancho), 0), (int(linea * ancho), alto)
        cv2.line(frame, p1, p2, AMARILLO, 2)
        cv2.putText(frame, nombre, (p1[0] + 8, p1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, AMARILLO, 2)

    for v in resultado['vehiculos']:
        caja = [v['caja'][0] * ancho, v['caja'][1] * alto, v['caja'][2] * ancho, v['caja'][3] * alto]
        texto = f"#{v['id']}"
        if v['velocidad'] is not None:
            texto += f" {v['velocidad']:.0f} km/h"
        elif v['midiendo']:
            texto += " midiendo"
        if v['placa']:
            texto += f" {v['placa']}"
        dibujar_vehiculo(frame, caja, ROJO if v['exceso'] else VERDE, texto)

    multas = len(resultado['multas'])
    cv2.putText(frame, f"Limite {config.limite_kmh:.0f} km/h | multas: {multas}", (20, alto - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLANCO, 2)
    return frame


# =============================================================================
# PROGRAMA PRINCIPAL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecta autos con exceso de velocidad en un video')
    parser.add_argument('video')
    parser.add_argument('--salida', default=None, help='video de salida (por defecto <video>_resultado.mp4)')
    parser.add_argument('--orientacion', choices=['vertical', 'horizontal'], default='vertical',
                        help='vertical: los autos pasan de lado; horizontal: vienen de frente')
    parser.add_argument('--linea-a', type=float, default=0.3, help='posición de la línea A (0 a 1)')
    parser.add_argument('--linea-b', type=float, default=0.7, help='posición de la línea B (0 a 1)')
    parser.add_argument('--distancia', type=float, default=10.0, help='distancia real entre las líneas (m)')
    parser.add_argument('--limite', type=float, default=60.0, help='límite de velocidad (km/h)')
    parser.add_argument('--paso', type=float, default=0.0,
                        help='segundos entre cuadros procesados (0 = todos los cuadros)')
    args = parser.parse_args()

    config = Configuracion(args.orientacion, args.linea_a, args.linea_b, args.distancia, args.limite)
    sesion = Sesion(DetectorExcesoVelocidad(), config)

    # load video (main.py, línea 18)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    salida = args.salida or os.path.splitext(args.video)[0] + '_resultado.mp4'
    fps_salida = fps if args.paso <= 0 else 1 / args.paso
    out = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*'mp4v'), fps_salida, (ancho, alto))

    # read frames (main.py, líneas 22-27)
    t_inicio = time.time()
    frame_nmr = -1
    siguiente = 0.0
    procesados = 0
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_nmr / fps   # tiempo del cuadro dentro del video
        if t + 1e-6 < siguiente:
            continue
        siguiente = t + args.paso

        resultado = sesion.procesar(frame, t)
        out.write(dibujar(frame, resultado, config))
        procesados += 1

    out.release()
    cap.release()

    print(f"\nCuadros procesados: {procesados} en {time.time() - t_inicio:.1f} s")
    print(f"Video con resultados: {salida}")
    print(f"\nMultas ({len(sesion.multas)}):")
    for m in sesion.multas:
        print(f"  {m['folio']}  placa {m['placa'] or '(no leída)':10s}  {m['velocidad_kmh']} km/h "
              f"(límite {m['limite_kmh']}, exceso {m['exceso_kmh']})")
    print(f"Registro: {sesion.detector.registro.ruta_csv}")
