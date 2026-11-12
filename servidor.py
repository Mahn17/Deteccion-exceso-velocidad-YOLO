#Manuel Alejandro Heredia Nogales
"""Servidor para usar el detector desde el celular

El celular abre la página (templates/index.html), toma cuadros con la cámara (o de un
video) y los envía a /api/procesar; el servidor responde con los vehículos, su
velocidad y las multas.

La estructura es la de app.py del notebook automatic-number-plate-recognition.ipynb
(celda 48): Flask + render_template + carpeta templates/.

Uso:
    python servidor.py
    ngrok http 5000      (en otra terminal; abrir en el celular la liga https://... de ngrok)
"""

# =============================================================================
# IMPORTACIONES

import json
import threading
import time

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from deteccion import DetectorExcesoVelocidad, Sesion

# =============================================================================
# CONFIGURACIÓN

PUERTO = 5000
SESION_INACTIVA = 600   # segundos sin recibir cuadros antes de borrar una sesión

# Campos de la configuración que puede mandar el celular y su validación
CAMPOS = {
    'orientacion': lambda v: v if v in ('vertical', 'horizontal') else None,
    'linea_a':     lambda v: float(v) if 0.02 <= float(v) <= 0.98 else None,
    'linea_b':     lambda v: float(v) if 0.02 <= float(v) <= 0.98 else None,
    'distancia_m': lambda v: float(v) if 0 < float(v) <= 500 else None,
    'limite_kmh':  lambda v: float(v) if 15 <= float(v) <= 300 else None,
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024   # 8 MB por cuadro

detector = DetectorExcesoVelocidad()
sesiones = {}                    # id -> [Sesion, último uso]
lock_sesiones = threading.Lock()


def obtener_sesion(id_):
    with lock_sesiones:
        ahora = time.time()
        for k in [k for k, (_, uso) in sesiones.items() if ahora - uso > SESION_INACTIVA]:
            del sesiones[k]
        if id_ not in sesiones:
            sesiones[id_] = [Sesion(detector), ahora]
        sesiones[id_][1] = ahora
        return sesiones[id_][0]


def leer_config(texto):
    try:
        datos = json.loads(texto)
        config = {campo: validar(datos[campo]) for campo, validar in CAMPOS.items() if campo in datos}
    except (ValueError, TypeError, AttributeError):
        abort(400, "La configuración no se pudo leer")
    for campo, valor in config.items():
        if valor is None:
            abort(400, f"Valor no válido para {campo}")
    if config.get('linea_a') is not None and config.get('linea_a') == config.get('linea_b'):
        abort(400, "Las líneas A y B deben estar en posiciones distintas")
    return config


# =============================================================================
# RUTAS

@app.route('/')
def index():
    return render_template('index.html')


@app.post('/api/procesar')
def procesar():
    archivo = request.files.get('imagen')
    if archivo is None or 't' not in request.form:
        abort(400, "Faltan la imagen o el tiempo")
    frame = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        abort(400, "La imagen no se pudo leer")

    sesion = obtener_sesion(request.form.get('sesion', 'default')[:64])
    if 'config' in request.form:
        sesion.configurar(**leer_config(request.form['config']))

    t0 = time.time()
    resultado = sesion.procesar(frame, float(request.form['t']))
    resultado['ms'] = round((time.time() - t0) * 1000)
    return jsonify(resultado)


@app.post('/api/reiniciar')
def reiniciar():
    with lock_sesiones:
        sesiones.pop(request.form.get('sesion', 'default')[:64], None)
    return jsonify(ok=True)


@app.get('/multas/<path:archivo>')
def imagen_multa(archivo):
    return send_from_directory(detector.registro.carpeta, archivo)


# =============================================================================
# PROGRAMA PRINCIPAL

if __name__ == '__main__':
    print("Cargando modelos (la primera vez se descargan yolov8n.pt y los modelos de EasyOCR)...")
    detector.calentar()
    print(f"\nListo. Abre http://localhost:{PUERTO} en esta computadora, o en otra terminal ejecuta")
    print(f"    ngrok http {PUERTO}")
    print("y abre en el celular la liga https://... que muestra ngrok.\n")
    # 127.0.0.1: solo se puede entrar desde esta computadora o a través de ngrok
    app.run(host='127.0.0.1', port=PUERTO, threaded=True)
