# 🚦 Detector de exceso de velocidad con YOLO

Aplicación web para el celular que detecta vehículos con YOLO, mide su velocidad y, si van
por encima del límite, genera una **multa simulada** con el número de placa.

Autor: Manuel Alejandro Heredia Nogales — Redes Neuronales 2026-2

> Simulación con fines educativos: las multas no tienen validez oficial.

---

## Cómo funciona

El celular solo muestra la imagen y dibuja los resultados; todo el análisis lo hace un
servidor en la computadora. Se comunican por internet a través de ngrok, que además da la
conexión HTTPS que el navegador exige para usar la cámara.

```
CELULAR (navegador)                              COMPUTADORA (servidor.py)
───────────────────                              ─────────────────────────
cámara o video
  └─ cuadro JPEG + tiempo t ───── ngrok ─────►  1. YOLOv8n (COCO): ¿dónde hay vehículos?
                                                 2. seguimiento: ¿quién es quién?
                                                 3. líneas A y B: ¿a qué velocidad va?
                                                 4. YOLOv8n de placas + EasyOCR: ¿qué placa?
                                                 5. ¿pasa el límite? → multa
dibuja cajas, velocidades  ◄───── JSON ────────
y la ventana de la multa
```

### 1. Del celular al servidor

La página (`templates/index.html`) muestra la cámara o el video en un elemento `<video>` y,
encima, un `<canvas>` transparente donde dibuja las líneas y las cajas.

Cada vez que el servidor termina con un cuadro, la página toma el cuadro que se está
mostrando, lo reduce a 1280 px de ancho como máximo, lo comprime en JPEG (unos 100 KB) y lo
envía a `/api/procesar` junto con:

- **`t`, el momento en que se tomó el cuadro.** En el modo Cámara es el reloj del celular;
  en el modo Video, el tiempo de ese cuadro dentro del video. Como la velocidad se calcula
  con el momento de captura y no con el de llegada, el retraso de la red no la altera.
- **`sesion`**, un identificador aleatorio de cada celular. El servidor lleva por separado
  el seguimiento y las multas de cada sesión, y la borra después de 10 minutos sin uso.
- **`config`**: la orientación y posición de las líneas, la distancia entre ellas y el límite.

Solo hay un cuadro en camino a la vez: así el servidor nunca se satura y los cuadros llegan
en orden.

### 2. Detección de vehículos: YOLOv8n con COCO

YOLO (*You Only Look Once*) encuentra todos los objetos de una imagen en una sola pasada de
la red neuronal:

1. La imagen se redimensiona para que su lado mayor mida 640 píxeles.
2. Una red convolucional extrae características a tres escalas, en cuadrículas de celdas de
   8, 16 y 32 píxeles: las celdas pequeñas sirven para objetos pequeños y las grandes para
   objetos grandes.
3. En cada celda, la red predice si hay un objeto, de cuál de las 80 clases de COCO es y la
   caja que lo rodea. YOLOv8 no usa cajas de referencia (*anchors*): predice directamente la
   distancia del centro de la celda a cada borde de la caja.
4. Se descartan las cajas con confianza menor a **0.35** y se aplica *non-max suppression*:
   de varias cajas muy traslapadas sobre el mismo objeto, se queda la de mayor confianza.
   Se hace sin importar la clase (`agnostic_nms`) para que un vehículo no salga dos veces,
   como "auto" y como "camión".
5. Se conservan solo las clases de vehículos: auto (2), moto (3), autobús (5) y camión (7).

Se usa **YOLOv8n** (*nano*), la versión más pequeña: 3.2 millones de parámetros y unos 8.7
GFLOPs por imagen, lo que permite usarla en CPU. Sus pesos los entrenó Ultralytics con COCO,
un dataset de unas 118 mil imágenes con 80 clases.

### 3. Seguimiento: ¿quién es quién?

YOLO analiza cada cuadro por separado: no sabe que la camioneta de este cuadro es la misma
del anterior. El seguimiento (clase `Rastreador` en `deteccion.py`) le da a cada vehículo un
número (`#1`, `#2`, …) que conserva mientras está en la imagen:

1. **Predicción.** Cada vehículo guarda su velocidad en píxeles por segundo (el promedio de
   la última medición y la anterior). Con ella se calcula dónde debería estar su caja en el
   momento del cuadro nuevo.
2. **Emparejamiento.** Para cada pareja (vehículo conocido, detección nueva) se compara la
   caja predicha con la detectada usando la **intersección sobre unión**:

   ```
   IoU = área de la intersección / área de la unión      (0 = no se tocan, 1 = idénticas)
   ```

   Con IoU de 0.10 o más, la pareja es candidata. Si las cajas no se traslapan, que es lo
   que pasa con autos rápidos y pocos cuadros por segundo, se usa la distancia entre sus
   centros: hasta 0.6 veces la diagonal de la caja, o 1.2 si el vehículo solo se ha visto
   una vez y todavía no se conoce su velocidad.
3. **Asignación.** Se asignan primero las parejas que más se traslapan. Cada vehículo y cada
   detección se usan una sola vez.
4. Las detecciones que quedan sin pareja son **vehículos nuevos**. Un vehículo que no se ve
   durante 1.5 s se olvida.
5. El tipo de vehículo se decide por votación entre cuadros, porque YOLO a veces duda entre
   auto y camión.

### 4. Velocidad: las líneas A y B

Una cámara no sabe cuántos metros hay en la imagen. Por eso se marcan dos líneas, A y B,
sobre dos puntos de la calle cuya distancia real se conoce. Es el mismo principio de los
radares de tramo: se mide el tiempo que tarda el vehículo en recorrer una distancia conocida.

- **Posición del vehículo.** Con líneas verticales (los autos pasan de lado) se usa el centro
  horizontal de su caja. Con líneas horizontales (los autos vienen de frente o se alejan) se
  usa el borde de abajo de la caja, donde las llantas tocan el pavimento, que es donde se
  midió la distancia.
- **Momento del cruce.** Los cuadros llegan cada cierto tiempo, así que casi nunca hay uno
  justo sobre la línea. Si en el momento `t0` el vehículo estaba en la posición `p0`, en `t1`
  en `p1`, y la línea `L` quedó entre las dos, el momento del cruce se interpola:

  ```
  t_cruce = t0 + (L − p0) / (p1 − p0) × (t1 − t0)
  ```

- **Velocidad.** Con los momentos de cruce de las dos líneas:

  ```
  velocidad (km/h) = distancia entre A y B (m) / |t_B − t_A| (s) × 3.6
  ```

  El vehículo puede ir de A a B o de B a A. Entre el primer cruce y el segundo, su etiqueta
  dice **"midiendo…"**.

La interpolación supone que el vehículo va a velocidad constante entre dos cuadros, lo cual
es razonable en tramos cortos. Las principales fuentes de error son una distancia mal
medida, un celular que se mueve y analizar muy pocos cuadros por segundo.

### 5. Lectura de la placa

1. **Recorte del vehículo** con un margen de 5 % alrededor de su caja.
2. **Detector de placas.** Es otra red YOLOv8n, entrenada para una sola clase,
   `license_plate`: se entrenó desde cero durante 30 épocas, con imágenes de 640 px, con un
   [dataset de placas de Roboflow](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).
   Buscar la placa dentro del recorte del vehículo, y no en toda la imagen, la hace verse
   más grande y permite encontrar placas pequeñas. Se acepta con confianza de 0.25 o más y
   se toma la de mayor confianza.
3. **Preparación.** Si la placa mide menos de 64 px de alto, se amplía. Se convierte a
   escala de grises y también se prueba una versión en blanco y negro (umbral de 64).
4. **EasyOCR** lee el texto con dos redes neuronales preentrenadas: una (CRAFT) localiza
   dónde hay texto y otra (convolucional con LSTM) reconoce los caracteres. Solo se aceptan
   letras mayúsculas, números y guiones.
5. **Unión de pedazos.** El OCR suele separar una placa como `ABC-123-D` en varios pedazos y
   leer también textos pequeños, como el nombre del estado. Se unen, de izquierda a derecha,
   los pedazos que miden al menos 60 % de la altura del más alto.
6. **Formato.** Se revisa que el texto tenga forma de placa (L = letra, N = número):
   `LLLNNNL`, `LLLNNNN`, `LLNNNNL` o `LLNNLLL`. Según la posición se corrigen las confusiones
   típicas del OCR: `O`↔`0`, `I`↔`1`, `J`↔`3`, `A`↔`4`, `G`↔`6` y `S`↔`5`. Si el texto cabe
   en varios formatos, se elige el que necesita menos correcciones.
7. **Votación entre cuadros.** Una sola lectura puede equivocarse aunque tenga mucha
   confianza, así que cada lectura suma su confianza al texto leído, y gana el texto con
   formato válido y mayor suma. Cuando un texto válido junta 3 lecturas, ese vehículo ya no
   se vuelve a leer.
8. **Costo.** El OCR es la parte más lenta, así que en cada cuadro se leen como máximo 2
   vehículos: primero los que ya tienen multa y luego los más grandes, que son los más
   cercanos a la cámara.

### 6. Multa

Si la velocidad medida supera el límite (que puede ser de 15 km/h o más), se genera una
multa con:

- folio consecutivo (`MS-0001`, `MS-0002`, …) y fecha y hora;
- placa, tipo de vehículo, velocidad, límite y exceso;
- monto simulado de $1,000;
- foto del vehículo, con la velocidad marcada, y foto de la placa.

Se guardan en `multas/multas.csv` junto con las fotos. Si la placa se lee después de crear la
multa, se completa en ese momento. En el celular aparece la ventana de la multa y se agrega a
la lista de multas generadas.

### 7. Lo que se ve en el celular

- **El video siempre se ve en vivo.** El análisis va aparte: el lienzo transparente de encima
  se redibuja en cada cuadro de la pantalla.
- **Las cajas acompañan a los autos.** Por ngrok, el servidor responde alrededor de una vez
  por segundo. Entre una respuesta y la siguiente, el celular mueve cada caja con la
  velocidad que calculó el servidor: `caja + velocidad × (ahora − t)`.
- **Colores:** azul mientras el vehículo no se ha medido, verde si va dentro del límite y rojo
  si lo excede.
- **Modo Video a 0.5×.** El video se reproduce a la mitad de su velocidad: se ve fluido y se
  alcanzan a analizar el doble de cuadros por cada segundo de video, lo que hace confiable
  la medición de autos rápidos. La velocidad medida no cambia, porque se calcula con el
  tiempo de cada cuadro dentro del video.

---

## Prueba

Para probar el sistema se armó un video sintético con velocidades conocidas, a partir de una
foto de una calle (`recursos/non-max-suppression.png`):

- Dos copias de una camioneta cruzan la calle a velocidad constante, cada una con una placa
  dibujada.
- Las líneas A y B están a 20 m; el límite es de 60 km/h.

```bash
python generar_video_prueba.py        # crea video_prueba.mp4 (6.5 s)
```

Para probarlo en el celular:
1. Pásalo al celular, por ejemplo por WhatsApp, Drive o con cable.
2. En la app, en **⚙ Calibración**, elige **De lado**, línea A en **30 %**, línea B en
   **70 %**, distancia **20 m** y límite **60 km/h**.
3. Pulsa **🎞 Video** y elige el archivo.

Debe multar a `VSA123B`, que va a 90 km/h, y no a `WKB4521`, que va a 45 km/h. Para que el
celular pueda reproducir el video (H.264), el script usa `ffmpeg`.

**Resultados**

En la computadora, con `procesar_video.py`, analizando distintos cuadros por segundo:

| Cuadros analizados por segundo | `VSA123B` (90 km/h) | `WKB4521` (45 km/h) |
|---|---|---|
| 30 | 90.1 km/h, con multa | sin multa |
| 10 | 90.0 km/h, con multa | sin multa |
| 3 | 89.9 km/h, con multa | sin multa |
| 2 | 88.1 km/h, con multa | sin multa |

En un navegador con pantalla de celular (390×844), con conexión local y con una conexión
lenta simulada parecida a ngrok (250 ms de latencia y 2 Mbps de subida):

| Modo | Conexión | Cuadros analizados por segundo | Multa a `VSA123B` |
|---|---|---|---|
| Video (0.5×) | local | 12–16 | 90.1 km/h |
| Video (0.5×) | lenta | 1.5 | 90.0 y 89.9 km/h |
| Cámara (simulada) | local | 20 | 90.1 km/h |
| Cámara (simulada) | lenta | 1.4 | 87.3 km/h |

En todos los casos el video se vio fluido. Con la predicción de movimiento en el celular, el
centro de cada caja quedó en promedio a 42–61 px del auto real (en una imagen de 1280 px),
contra 112–187 px sin predicción.

Todavía falta probarla con tráfico real.

---

## Estructura

```
├── servidor.py                 # servidor Flask: recibe los cuadros del celular
├── deteccion.py                # vehículos, seguimiento, velocidad, placas y multas
├── placas.py                   # lectura de placas con EasyOCR
├── procesar_video.py           # procesa un video en la computadora (sin celular)
├── generar_video_prueba.py     # crea video_prueba.mp4, con velocidades conocidas
├── templates/index.html        # la aplicación del celular
├── license_plate_detector.pt   # detector de placas (YOLOv8n)
├── recursos/                   # foto para el video de prueba
├── requirements.txt
└── multas/                     # se crea al generar multas: multas.csv y fotos
```

`yolov8n.pt` se descarga solo la primera vez que se ejecuta.

---

## Instalación

Probado con Python 3.12 y 3.14, sin GPU.

```bash
git clone https://github.com/Mahn17/Deteccion-exceso-velocidad-YOLO.git
cd Deteccion-exceso-velocidad-YOLO
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

La primera ejecución descarga `yolov8n.pt` (6.5 MB) y los modelos de EasyOCR (unos 100 MB).

---

## Uso desde el celular

```bash
# terminal 1
source .venv/bin/activate
python servidor.py

# terminal 2
ngrok http 5000
```

Abre en el celular la liga `https://….ngrok-free.app` y pulsa **Visit Site**.

**Calibración:**
1. Fija el celular, por ejemplo en un tripié. Si se mueve, la medición no sirve.
2. Elige cómo pasan los autos:
   - **De lado:** las líneas son verticales.
   - **De frente o de espalda:** las líneas son horizontales.
3. Coloca las líneas A y B sobre dos marcas de la calle y escribe la distancia real entre
   ellas, por ejemplo dos postes o dos marcas del pavimento medidas con cinta. Entre más
   separadas, mejor, siempre que el vehículo se vea completo al cruzar las dos.
4. Escribe el límite de velocidad (mínimo 15 km/h). La distancia y el límite se aplican al
   terminar de escribir.

**Modos:**

- **📷 Cámara:** analiza en vivo. En la computadora, sin ngrok, se procesan de 13 a 20
  cuadros por segundo; por ngrok, alrededor de 1.5, porque lo que más tarda es subir cada
  imagen. Como la calle no se puede poner en cámara lenta, conviene que los autos tarden en
  cruzar la pantalla: el celular más lejos de la calle y las líneas bien separadas.
- **🎞 Video:** reproduce un video grabado o elegido en el celular, a 0.5× de velocidad, y
  analiza el cuadro que se esté mostrando cada vez que el servidor termina con el anterior.

---

## Procesar un video en la computadora

```bash
python procesar_video.py autos.mp4 --distancia 20 --limite 60
python procesar_video.py autos.mp4 --orientacion horizontal --linea-a 0.55 --linea-b 0.85 --paso 0.1
```

Genera `autos_resultado.mp4` con las cajas, las velocidades y las placas, e imprime las
multas.

---

## Ajustes

| Qué | Dónde |
|---|---|
| Formatos de placa aceptados (L = letra, N = número) | `FORMATOS_PLACA` en `placas.py` |
| Monto de la multa simulada | `MONTO_MULTA` en `deteccion.py` |
| Clases de vehículos | `vehicles` en `deteccion.py` |
| Límite mínimo | `LIMITE_MIN` en `templates/index.html` y `CAMPOS` en `servidor.py` |
| Velocidad de reproducción del modo Video | `VELOCIDAD_VIDEO` en `templates/index.html` |
| Tamaño de los cuadros que se envían (menor = más rápido por ngrok, pero placas más pequeñas) | `ANCHO_MAX` en `templates/index.html` |

Para reiniciar la numeración de los folios, borra la carpeta `multas/`.

---

## Créditos y licencias

- **Detector de vehículos:** [YOLOv8n de Ultralytics](https://github.com/ultralytics/ultralytics)
  (licencia AGPL-3.0), preentrenado con [COCO](https://cocodataset.org).
- **Detector de placas** (`license_plate_detector.pt`) y funciones de lectura de placas
  adaptadas en `placas.py`: proyecto
  [Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8),
  © 2023 Muhammad Zeerak Khan, licencia MIT (ver `LICENSE-ANPR`).
- **Lectura de texto:** [EasyOCR](https://github.com/JaidedAI/EasyOCR), de JaidedAI
  (licencia Apache-2.0).
- **Foto del video de prueba:** figura del curso *Deep Learning Specialization* de
  DeepLearning.AI en Coursera, con una foto del dataset de drive.ai.

---

## Limitaciones

- La velocidad es tan buena como la calibración: la distancia real entre las líneas y un
  celular fijo.
- Con pocos cuadros por segundo, un auto rápido que cruza la pantalla muy rápido puede no
  medirse.
- Las placas lejanas, pequeñas o de formatos distintos a los configurados pueden no leerse.
