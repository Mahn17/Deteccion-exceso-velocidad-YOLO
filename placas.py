#Manuel Alejandro Heredia Nogales
"""Lectura de placas con EasyOCR

Adaptado de py/yolo/Automatic-License-Plate-Recognition-using-YOLOv8-main/util.py:
  * dict_char_to_int / dict_int_to_char       -> iguales (líneas 8-20)
  * license_complies_format / format_license  -> generalizadas a varios formatos
  * read_license_plate                        -> leer_placa (también une los pedazos de texto)

util.py solo acepta el formato británico de 7 caracteres (2 letras, 2 números,
3 letras). Aquí el formato se describe con una cadena: L = letra, N = número.
"""

# =============================================================================
# IMPORTACIONES

import string

import cv2
import easyocr

# =============================================================================
# CONFIGURACIÓN

# Formatos de placa aceptados (L = letra, N = número). Ajustar a las placas locales.
FORMATOS_PLACA = ["LLLNNNL", "LLLNNNN", "LLNNNNL", "LLNNLLL"]

# Si el texto no coincide con ningún formato, se acepta igual si tiene entre 5 y 8
# caracteres con al menos una letra y un número (se marca como formato no válido)
LONGITUD_MIN, LONGITUD_MAX = 5, 8

# Diccionarios para corregir confusiones del OCR (util.py, líneas 8-20)
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

DIGITOS = '0123456789'

# El lector OCR se crea la primera vez que se usa (util.py lo creaba al importar).
# La primera vez EasyOCR descarga sus modelos (~100 MB).
_reader = None


def lector():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


# =============================================================================
# FORMATO DE LA PLACA

def license_complies_format(text, formato):
    """True si cada carácter es del tipo pedido (L/N) o se puede convertir a él."""
    if len(text) != len(formato):
        return False

    for c, tipo in zip(text, formato):
        if tipo == 'L' and not (c in string.ascii_uppercase or c in dict_int_to_char.keys()):
            return False
        if tipo == 'N' and not (c in DIGITOS or c in dict_char_to_int.keys()):
            return False
    return True


def format_license(text, formato):
    """Convierte los caracteres confundidos según la posición (util.py, format_license)."""
    license_plate_ = ''
    for c, tipo in zip(text, formato):
        mapping = dict_int_to_char if tipo == 'L' else dict_char_to_int
        license_plate_ += mapping.get(c, c)
    return license_plate_


def validar_placa(text):
    """Regresa (placa, formato_valido) o (None, False) si el texto no parece una placa.

    Si el texto cumple con varios formatos se usa el que requiere menos correcciones
    (por ejemplo, "WKB4521" es LLLNNNN tal cual; como LLLNNNL habría que cambiar el 1 por I).
    """
    text = ''.join(c for c in text.upper() if c.isalnum())

    formatos = [f for f in FORMATOS_PLACA if license_complies_format(text, f)]
    if formatos:
        formato = min(formatos, key=lambda f: sum(a != b for a, b in zip(text, format_license(text, f))))
        return format_license(text, formato), True

    tiene_letra = any(c in string.ascii_uppercase for c in text)
    tiene_numero = any(c in DIGITOS for c in text)
    if LONGITUD_MIN <= len(text) <= LONGITUD_MAX and tiene_letra and tiene_numero:
        return text, False

    return None, False


# =============================================================================
# LECTURA DE LA PLACA

def _candidatos(detecciones):
    """Textos a probar: cada detección sola y los pedazos grandes unidos de izquierda a derecha.

    En una placa mexicana el OCR suele separar "ABC-123-D" en varios pedazos, y además
    lee el texto pequeño (nombre del estado). Para unir solo los caracteres principales
    se toman los pedazos con al menos 60 % de la altura del más alto.
    """
    candidatos = [(text, score) for _, text, score in detecciones]

    alturas = [max(p[1] for p in bbox) - min(p[1] for p in bbox) for bbox, _, _ in detecciones]
    if len(detecciones) > 1:
        altura_max = max(alturas)
        grandes = [(min(p[0] for p in bbox), text, score)
                   for (bbox, text, score), h in zip(detecciones, alturas) if h >= 0.6 * altura_max]
        if len(grandes) > 1:
            grandes.sort()
            candidatos.append((''.join(t for _, t, _ in grandes),
                               sum(s for _, _, s in grandes) / len(grandes)))
    return candidatos


def leer_placa(license_plate_crop):
    """Lee el texto de un recorte de placa (BGR).

    Regresa (placa, score, formato_valido) o (None, None, False).
    Como en main.py (líneas 55-56) se prueba también la versión umbralizada.
    """
    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None, False

    # EasyOCR lee mejor si la placa mide al menos ~64 píxeles de alto
    alto = license_plate_crop.shape[0]
    if alto < 64:
        escala = 64 / alto
        license_plate_crop = cv2.resize(license_plate_crop, None, fx=escala, fy=escala,
                                        interpolation=cv2.INTER_CUBIC)

    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

    mejor = (None, None, False)
    for imagen in (license_plate_crop_gray, license_plate_crop_thresh):
        detections = lector().readtext(imagen, allowlist=string.ascii_uppercase + DIGITOS + '-')
        for text, score in _candidatos(detections):
            placa, valido = validar_placa(text)
            if placa is None:
                continue
            # Se prefiere un formato válido y, después, la mayor confianza
            if mejor[0] is None or (valido, score) > (mejor[2], mejor[1]):
                mejor = (placa, score, valido)
        if mejor[2]:
            break

    return mejor
