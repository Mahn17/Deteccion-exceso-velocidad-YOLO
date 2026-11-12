#Manuel Alejandro Heredia Nogales
"""Detección de autos con exceso de velocidad: YOLO + seguimiento + placas

Flujo por cuadro, basado en py/yolo/Automatic-License-Plate-Recognition-using-YOLOv8-main/main.py:
  1. YOLO (yolov8n, preentrenado con COCO) detecta vehículos (main.py, líneas 20-36).
  2. Seguimiento: cada vehículo conserva su id entre cuadros. main.py usa SORT, que no
     viene en el repositorio (se descarga de github.com/abewley/sort); aquí se usa un
     seguimiento sencillo basado en iou() de v1_autonomous_driving_application_car_detection.py.
  3. Velocidad: dos líneas virtuales separadas por una distancia real conocida.
        velocidad = distancia / (momento en que cruza la línea B - momento en que cruza la A)
  4. Placa: license_plate_detector.pt (mismo modelo de main.py) busca la placa dentro del
     recorte de cada vehículo y EasyOCR la lee (placas.py).
  5. Si la velocidad supera el límite se genera una multa simulada.
"""

# =============================================================================
# IMPORTACIONES

import csv
import datetime
import os
import threading
import zoneinfo
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from ultralytics import YOLO

import placas

# =============================================================================
# CONFIGURACIÓN

DIR_BASE = os.path.dirname(os.path.abspath(__file__))

# Clases de COCO que son vehículos: 2 car, 3 motorcycle, 5 bus, 7 truck (main.py, línea 20)
vehicles = [2, 3, 5, 7]
NOMBRES_VEHICULOS = {2: 'auto', 3: 'moto', 5: 'autobús', 7: 'camión'}

CONF_VEHICULOS = 0.35   # confianza mínima para aceptar un vehículo
CONF_PLACAS    = 0.25   # confianza mínima para aceptar una placa
IOU_MIN        = 0.10   # IoU mínimo para considerar que dos cajas son el mismo vehículo
DISTANCIA_MAX  = 0.6    # si no se traslapan: distancia máxima entre centros (fracción de la diagonal)
DISTANCIA_MAX_NUEVO = 1.2  # lo mismo para un vehículo visto una sola vez, cuya velocidad aún no se conoce
LECTURAS_PLACA = 3      # lecturas que coinciden para dejar de leer la placa de un vehículo
EDAD_MAX       = 1.5    # segundos sin ver un vehículo antes de olvidarlo
OCR_POR_CUADRO = 2      # vehículos a los que se les lee la placa en cada cuadro
MONTO_MULTA    = 1000   # pesos (simulado)

zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")


@dataclass
class Configuracion:
    orientacion: str = 'vertical'   # 'vertical': líneas verticales, los autos pasan de lado
                                    # 'horizontal': líneas horizontales, los autos vienen de frente
    linea_a: float = 0.3            # posición de las líneas (fracción del ancho o del alto)
    linea_b: float = 0.7
    distancia_m: float = 10.0       # distancia real entre las dos líneas
    limite_kmh: float = 60.0


# =============================================================================
# FUNCIONES DE APOYO

def iou(box1, box2):
    """Intersección sobre unión (v1_autonomous_driving_application_car_detection.py, líneas 571-612)."""
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1

    if inter_height <= 0 or inter_width <= 0:
        inter_area = 0
    else:
        inter_area = inter_width * inter_height

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """Esquinas de la caja de un vehículo (visualize.py, líneas 8-24)."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def dibujar_vehiculo(img, caja, color, texto=None):
    """draw_border con el grosor y el largo de las esquinas proporcionales a la caja."""
    x1, y1, x2, y2 = (int(v) for v in caja)
    grosor = max(2, round(max(img.shape[:2]) / 400))
    draw_border(img, (x1, y1), (x2, y2), color, grosor,
                line_length_x=max(8, (x2 - x1) // 4), line_length_y=max(8, (y2 - y1) // 4))
    if texto:
        escala = max(0.6, max(img.shape[:2]) / 1200)
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
        x_texto = max(0, min(x1, img.shape[1] - w - 8))   # que la etiqueta no se salga de la imagen
        y_texto = y1 - 8 if y1 - h - 12 > 0 else y2 + h + 8
        cv2.rectangle(img, (x_texto, y_texto - h - 6), (x_texto + w + 8, y_texto + 6), color, -1)
        cv2.putText(img, texto, (x_texto + 4, y_texto), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2)
    return img


def centro(caja):
    return (caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2


def coordenada(caja, ancho, alto, orientacion):
    """Posición del vehículo (0 a 1) en la dirección en que se mide la velocidad.

    Líneas verticales (los autos pasan de lado): el centro horizontal de la caja.
    Líneas horizontales (los autos vienen de frente): la parte de abajo de la caja,
    que es donde las llantas tocan el piso.
    """
    if orientacion == 'horizontal':
        return caja[3] / alto
    return centro(caja)[0] / ancho


# =============================================================================
# SEGUIMIENTO DE VEHÍCULOS

class Vehiculo:
    def __init__(self, id_, caja, clase, t):
        self.id = id_
        self.caja = caja                # [x1, y1, x2, y2] en píxeles
        self.clases = {clase: 1}        # votos por clase (YOLO a veces confunde auto y camión)
        self.t = t                      # último momento en que se vio (segundos)
        self.vel = (0.0, 0.0)           # velocidad del centro en píxeles/s, para predecir su posición
        self.vistas = 1
        self.cruces = {}                # {'A': t, 'B': t}: momento en que cruzó cada línea
        self.velocidad_kmh = None
        self.lecturas = {}              # {placa: {'votos', 'suma', 'valida', 'score', 'recorte'}}
        self.placa = None
        self.placa_score = 0.0
        self.placa_recorte = None
        self.multa = None

    def agregar_lectura(self, texto, score, valida, recorte):
        """Suma una lectura de la placa. Regresa True si cambió la placa elegida.

        visualize.py elige la lectura con mayor confianza; aquí cada lectura es un voto
        (pesado por su confianza), porque una sola lectura equivocada puede tener mucha confianza.
        """
        l = self.lecturas.setdefault(texto, {'votos': 0, 'suma': 0.0, 'valida': valida, 'score': 0.0, 'recorte': None})
        l['votos'] += 1
        l['suma'] += score
        if score > l['score']:
            l['score'], l['recorte'] = score, recorte

        mejor = max(self.lecturas, key=lambda p: (self.lecturas[p]['valida'], self.lecturas[p]['suma']))
        cambio = mejor != self.placa
        self.placa = mejor
        self.placa_score = self.lecturas[mejor]['score']
        self.placa_recorte = self.lecturas[mejor]['recorte']
        return cambio or self.lecturas[mejor]['recorte'] is recorte

    @property
    def placa_confirmada(self):
        return self.placa is not None and self.lecturas[self.placa]['valida'] and \
            self.lecturas[self.placa]['votos'] >= LECTURAS_PLACA

    @property
    def clase(self):
        return max(self.clases, key=self.clases.get)

    def prediccion(self, t):
        """Caja desplazada según su velocidad: ayuda a seguir autos rápidos con pocos cuadros por segundo."""
        dt = t - self.t
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        x1, y1, x2, y2 = self.caja
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def mover(self, caja, clase, t):
        dt = t - self.t
        if dt > 0:
            (cx0, cy0), (cx1, cy1) = centro(self.caja), centro(caja)
            medida = ((cx1 - cx0) / dt, (cy1 - cy0) / dt)
            if self.vistas == 1:
                self.vel = medida
            else:
                self.vel = (0.5 * medida[0] + 0.5 * self.vel[0], 0.5 * medida[1] + 0.5 * self.vel[1])
        self.caja = caja
        self.t = t
        self.vistas += 1
        self.clases[clase] = self.clases.get(clase, 0) + 1


class Rastreador:
    """Asigna las detecciones de cada cuadro a los vehículos que ya se venían siguiendo."""

    def __init__(self):
        self.vehiculos = {}
        self.siguiente_id = 1

    def actualizar(self, detecciones, t):
        """detecciones: [x1, y1, x2, y2, score, class_id].

        Regresa [(vehiculo, caja_anterior, t_anterior)] de los vehículos vistos en este
        cuadro; caja_anterior es None si el vehículo es nuevo.
        """
        # Olvidar los vehículos que ya no se ven
        self.vehiculos = {i: v for i, v in self.vehiculos.items() if t - v.t <= EDAD_MAX}

        # Posibles parejas (vehículo, detección). Primero las que se traslapan (IoU) y,
        # si no se traslapan, las que tienen el centro cerca de donde se esperaba el vehículo.
        pares = []
        for v in self.vehiculos.values():
            prediccion = v.prediccion(t)
            cx, cy = centro(prediccion)
            for j, d in enumerate(detecciones):
                s = iou(prediccion, d[:4])
                if s >= IOU_MIN:
                    pares.append((1 + s, v.id, j))
                    continue
                dx, dy = centro(d[:4])
                diagonal = np.hypot(d[2] - d[0], d[3] - d[1])
                dist = np.hypot(dx - cx, dy - cy) / diagonal
                maxima = DISTANCIA_MAX_NUEVO if v.vistas == 1 else DISTANCIA_MAX
                if dist < maxima:
                    pares.append((1 - dist / maxima, v.id, j))

        pares.sort(reverse=True)
        vehiculos_usados, detecciones_usadas = set(), set()
        vistos = []
        for _, id_, j in pares:
            if id_ in vehiculos_usados or j in detecciones_usadas:
                continue
            vehiculos_usados.add(id_)
            detecciones_usadas.add(j)
            v = self.vehiculos[id_]
            anterior = (v.caja, v.t)
            v.mover(detecciones[j][:4], detecciones[j][5], t)
            vistos.append((v, *anterior))

        # Las detecciones que sobran son vehículos nuevos
        for j, d in enumerate(detecciones):
            if j not in detecciones_usadas:
                v = Vehiculo(self.siguiente_id, d[:4], d[5], t)
                self.siguiente_id += 1
                self.vehiculos[v.id] = v
                vistos.append((v, None, None))

        return vistos


# =============================================================================
# REGISTRO DE MULTAS (multas/multas.csv, como write_csv de util.py)

class RegistroMultas:
    CAMPOS = ['folio', 'fecha', 'placa', 'placa_confianza', 'tipo', 'velocidad_kmh',
              'limite_kmh', 'exceso_kmh', 'monto', 'imagen_auto', 'imagen_placa']

    def __init__(self, carpeta):
        self.carpeta = carpeta
        os.makedirs(carpeta, exist_ok=True)
        self.ruta_csv = os.path.join(carpeta, 'multas.csv')
        self.lock = threading.Lock()
        self.multas = []
        if os.path.exists(self.ruta_csv):
            with open(self.ruta_csv, newline='', encoding='utf-8') as f:
                self.multas = list(csv.DictReader(f))

    def nueva(self, datos, imagen_auto):
        with self.lock:
            folio = f"MS-{len(self.multas) + 1:04d}"
            multa = {campo: '' for campo in self.CAMPOS}
            multa.update(datos, folio=folio, imagen_auto=f"{folio}_auto.jpg")
            cv2.imwrite(os.path.join(self.carpeta, multa['imagen_auto']), imagen_auto)
            self.multas.append(multa)
            self._guardar()
            return multa

    def actualizar_placa(self, multa, placa, score, recorte):
        with self.lock:
            multa['placa'] = placa
            multa['placa_confianza'] = round(float(score), 3)
            multa['imagen_placa'] = f"{multa['folio']}_placa.jpg"
            cv2.imwrite(os.path.join(self.carpeta, multa['imagen_placa']), recorte)
            self._guardar()

    def _guardar(self):
        with open(self.ruta_csv, 'w', newline='', encoding='utf-8') as f:
            escritor = csv.DictWriter(f, fieldnames=self.CAMPOS)
            escritor.writeheader()
            escritor.writerows(self.multas)


# =============================================================================
# DETECTOR (modelos compartidos por todas las sesiones)

class DetectorExcesoVelocidad:
    def __init__(self, modelo_vehiculos=os.path.join(DIR_BASE, 'yolov8n.pt'),
                 modelo_placas=os.path.join(DIR_BASE, 'license_plate_detector.pt'),
                 carpeta_multas=os.path.join(DIR_BASE, 'multas')):
        # load models (main.py, líneas 13-15). yolov8n.pt se descarga solo la primera vez.
        self.coco_model = YOLO(modelo_vehiculos)
        self.license_plate_detector = YOLO(modelo_placas)
        self.registro = RegistroMultas(carpeta_multas)
        self.lock = threading.Lock()   # los modelos se usan de uno en uno

    def calentar(self):
        """Primera inferencia de cada modelo y del OCR, para que el primer cuadro no tarde."""
        negro = np.zeros((480, 640, 3), dtype=np.uint8)
        self.coco_model(negro, verbose=False)
        self.license_plate_detector(negro, verbose=False)
        placas.lector().readtext(negro[:64, :200])

    def detectar_vehiculos(self, frame):
        """detect vehicles (main.py, líneas 30-36). agnostic_nms evita que el mismo vehículo
        salga dos veces, como 'car' y como 'truck'."""
        detections = self.coco_model(frame, conf=CONF_VEHICULOS, agnostic_nms=True, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score, int(class_id)])
        return detections_

    def leer_placa(self, frame, caja):
        """Busca la placa dentro del vehículo y la lee. Regresa (placa, score, valida, recorte) o None.

        main.py busca placas en todo el cuadro y luego las asigna a cada auto con get_car();
        buscar dentro del recorte del auto equivale a eso y detecta mejor las placas pequeñas.
        """
        alto, ancho = frame.shape[:2]
        x1, y1, x2, y2 = caja
        mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))
        auto = frame[y1:y2, x1:x2]
        if auto.shape[0] < 32 or auto.shape[1] < 32:
            return None

        license_plates = self.license_plate_detector(auto, conf=CONF_PLACAS, verbose=False)[0]
        if len(license_plates.boxes) == 0:
            return None
        px1, py1, px2, py2, _, _ = max(license_plates.boxes.data.tolist(), key=lambda b: b[4])

        # crop license plate (main.py, línea 52)
        license_plate_crop = auto[int(py1):int(py2), int(px1):int(px2), :]
        texto, texto_score, valida = placas.leer_placa(license_plate_crop)
        if texto is None:
            return None
        return texto, texto_score, valida, license_plate_crop


# =============================================================================
# SESIÓN (un celular o un video)

class Sesion:
    def __init__(self, detector, config=None):
        self.detector = detector
        self.config = config or Configuracion()
        self.rastreador = Rastreador()
        self.multas = []
        self.t_ultimo = None

    def configurar(self, **datos):
        """Cambia la configuración. Si se mueven las líneas se reinician las mediciones en curso."""
        nueva = Configuracion(**{**asdict(self.config), **datos})
        lineas = ('orientacion', 'linea_a', 'linea_b', 'distancia_m')
        if any(getattr(nueva, c) != getattr(self.config, c) for c in lineas):
            for v in self.rastreador.vehiculos.values():
                v.cruces.clear()
        self.config = nueva

    def _medir(self, v, p0, t0, p1, t1):
        """Registra el momento exacto en que el vehículo cruza cada línea.

        Entre dos cuadros el cruce se interpola: si en t0 estaba en p0 y en t1 en p1,
        cruzó la línea L en t0 + (L - p0) / (p1 - p0) * (t1 - t0).
        """
        c = self.config
        for nombre, linea in (('A', c.linea_a), ('B', c.linea_b)):
            if nombre not in v.cruces and (p0 < linea <= p1 or p0 > linea >= p1):
                v.cruces[nombre] = t0 + (linea - p0) / (p1 - p0) * (t1 - t0)

        if v.velocidad_kmh is None and len(v.cruces) == 2:
            dt = abs(v.cruces['B'] - v.cruces['A'])
            if dt > 0.05:
                v.velocidad_kmh = round(c.distancia_m / dt * 3.6, 1)

    def _crear_multa(self, v, frame):
        x1, y1, x2, y2 = (int(n) for n in v.caja)
        alto, ancho = frame.shape[:2]
        mx, my = (x2 - x1) // 2, (y2 - y1) // 2
        imagen = frame[max(0, y1 - my):min(alto, y2 + my), max(0, x1 - mx):min(ancho, x2 + mx)].copy()
        caja_local = [x1 - max(0, x1 - mx), y1 - max(0, y1 - my), x2 - max(0, x1 - mx), y2 - max(0, y1 - my)]
        dibujar_vehiculo(imagen, caja_local, (0, 0, 255), f"{v.velocidad_kmh:.0f} km/h")

        limite = self.config.limite_kmh
        v.multa = self.detector.registro.nueva({
            'fecha': datetime.datetime.now(zona_hermosillo).strftime('%Y-%m-%d %H:%M:%S'),
            'tipo': NOMBRES_VEHICULOS.get(v.clase, 'vehículo'),
            'velocidad_kmh': v.velocidad_kmh,
            'limite_kmh': limite,
            'exceso_kmh': round(v.velocidad_kmh - limite, 1),
            'monto': MONTO_MULTA,
        }, imagen)
        if v.placa:
            self.detector.registro.actualizar_placa(v.multa, v.placa, v.placa_score, v.placa_recorte)
        self.multas.append(v.multa)

    def procesar(self, frame, t):
        """Procesa un cuadro tomado en el momento t (segundos). Regresa un diccionario para la app."""
        with self.detector.lock:
            # Si el tiempo regresa (por ejemplo, el video volvió a empezar) se reinicia el seguimiento
            if self.t_ultimo is not None and t < self.t_ultimo:
                self.rastreador = Rastreador()
            self.t_ultimo = t

            alto, ancho = frame.shape[:2]
            c = self.config
            vistos = self.rastreador.actualizar(self.detector.detectar_vehiculos(frame), t)

            # Velocidad y multas
            for v, caja_anterior, t_anterior in vistos:
                if caja_anterior is not None:
                    self._medir(v, coordenada(caja_anterior, ancho, alto, c.orientacion), t_anterior,
                                coordenada(v.caja, ancho, alto, c.orientacion), t)
                if v.velocidad_kmh is not None and v.velocidad_kmh > c.limite_kmh and v.multa is None:
                    self._crear_multa(v, frame)

            # Placas: primero los vehículos multados y luego los más grandes (más cerca de la cámara)
            pendientes = [v for v, _, _ in vistos if not v.placa_confirmada]
            pendientes.sort(key=lambda v: (v.multa is not None, (v.caja[2] - v.caja[0]) * (v.caja[3] - v.caja[1])),
                            reverse=True)
            for v in pendientes[:OCR_POR_CUADRO]:
                lectura = self.detector.leer_placa(frame, v.caja)
                if lectura is None:
                    continue
                if v.agregar_lectura(*lectura) and v.multa is not None:
                    self.detector.registro.actualizar_placa(v.multa, v.placa, v.placa_score, v.placa_recorte)

            return {
                'ancho': ancho,
                'alto': alto,
                't': t,
                'vehiculos': [{
                    'id': v.id,
                    'caja': [v.caja[0] / ancho, v.caja[1] / alto, v.caja[2] / ancho, v.caja[3] / alto],
                    # Para que el celular mueva la caja entre una respuesta y la siguiente
                    'vel': [v.vel[0] / ancho, v.vel[1] / alto],
                    'tipo': NOMBRES_VEHICULOS.get(v.clase, 'vehículo'),
                    'velocidad': v.velocidad_kmh,
                    'midiendo': len(v.cruces) == 1,
                    'exceso': v.velocidad_kmh is not None and v.velocidad_kmh > c.limite_kmh,
                    'placa': v.placa,
                } for v, _, _ in vistos],
                'multas': self.multas,
            }
