# ClearSky - Detección y Segmentación de Residuos

Proyecto para detectar y clasificar residuos (Cardboard, Organic, Plastic) mediante cámara cenital + LiDAR SLAMTEC M1M1, desplegado en Jetson Nano 4GB (JetPack 4.6.1).

## 📁 Estructura del repositorio

```
ClearSkypy/
├── pretrained/                    # Pesos base pre-entrenados
│   ├── yolov8n.pt                 # YOLOv8n detection (oficial Ultralytics)
│   └── yolov8n-seg.pt             # YOLOv8n segmentation (oficial Ultralytics)
├── datasets/                      # Datasets (fuera del repo, ver abajo)
│   └── Clear_v1.v2-clear_v1.1_-3-tags.yolov8/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
├── src/training/train/
│   ├── train_seg.py               # Entrenamiento YOLOv8 (detect o segment)
│   ├── test_model.py              # Prueba en vivo con cámara / archivo
│   ├── runs/                      # Salidas de entrenamiento (se crean al correr)
│   │   ├── YOLO_Detection/
│   │   └── YOLO_Segmentation/
│   ├── main.py                    # Entrenamiento legacy YOLO26s (referencia)
│   └── mlflow.db                  # Tracking de experimentos
└── README.md                      # Este archivo
```

## 🚀 Quick Start

### 1. Activar entorno virtual (fish shell)
```fish
source .venv/bin/activate.fish
```

### 2. Verificar cámara
```bash
ls /dev/video*
v4l2-ctl --list-formats-ext
```
La cámara USB (ELFO LARANJA / Sonix 0c49) soporta **MJPG 1920x1080 @ 30 fps**. Para pruebas usamos 1280x720.

### 3. Entrenar modelo (YOLOv8n detection)
```bash
python src/training/train/train_seg.py
```
- Usa `yolov8n.pt` (detección) + dataset `Clear_v1.v2-clear_v1.1_-3-tags.yolov8`.
- `task="detect"` para evitar error de etiquetas mezcladas.
- Pesos guardados en `src/training/train/runs/YOLO_Segmentation/yolo8n_detect_clear_sky/weights/best.pt`.

### 4. Probar modelo en vivo con cámara
```bash
python src/training/train/test_model.py
```
- Carga automáticamente el último `best.pt` entrenado (YOLOv8n detect).
- Abre cámara `/dev/video0` en MJPG 1280x720 @ 30 fps.
- Muestra FPS y conteo por clase (Cardboard / Organic / Plastic).
- Presiona `q` para salir.

## ⚙️ Parámetros clave a ajustar

| Parámetro | Script | Flag / Variable | Valor por defecto | Cuándo cambiar |
|-----------|--------|-----------------|-------------------|----------------|
| **Confidence threshold** | `test_model.py` | `--conf` | **0.55** | Sube para más precisión (menos falsos positivos), baja para más recall (menos falsos negativos). |
| **Batch size** | `train_seg.py` | `batch=` | 2 | Si `CUDA out of memory` → pon `batch=1`. |
| **Épocas** | `train_seg.py` | `epochs=` | 100 | Reduce para pruebas rápidas (ej. 5). |
| **Learning rate** | `train_seg.py` | `lr0=` | 0.002 | Funcionó bien en experimentos previos. |
| **Resolución inferencia** | `test_model.py` | `--imgsz` | 640 | Sube a 720/960 si los objetos son muy pequeños. |
| **Cámara** | `test_model.py` | `--camera` `--width` `--height` `--fps` | 0, 1280, 720, 30 | Si tienes varias cámaras o quieres otra resolución. |
| **Fuente de entrada** | `test_model.py` | `--source` | None (webcam) | Pasa ruta a video/imagen para probar sin cámara. |
| **Modelo a cargar** | `test_model.py` | (auto) | `best.pt` YOLOv8n detect | El script busca primero el modelo YOLOv8n detect actual; si no existe, cae al YOLO26s legacy. |

## 📊 Métricas de referencia (entrenamientos previos)

| Modelo | mAP50 | mAP50-95 | Organic Recall | Comentario |
|--------|-------|----------|----------------|------------|
| YOLO26s detect (legacy) | 0.383 | 0.245 | 0.249 | Modelo base, buen recall en Cardboard/Plastic, bajo en Organic. |
| YOLOv8n detect (actual) | *pendiente* | *pendiente* | *pendiente* | Entrenado con `task=detect` sobre dataset `.yolov8`. |

**Objetivo del proyecto:** Recall ≥ 0.8 en Organic (el LiDAR descarta falsos positivos).

## 🗂️ Dataset

- **Fuente:** Roboflow `clear_v1` v2 (3 clases: Cardboard, Organic, Plastic).
- **Formato:** YOLO segmentation (polígonos) + cajas. El dataset original trae etiquetas mixtas (boxes + segments) → por eso se usa `task=detect` para entrenar sin error.
- **Ubicación:** `/home/santiago/Proyectos/datasets/Clear_v1.v2-clear_v1.1_-3-tags.yolov8/` (fuera del repo por tamaño).
- **Split:** train 783 / valid 65 / test 46 imágenes (600x450).

## 🖥️ Despliegue en Jetson Nano (JetPack 4.6.1)

1. Instalar Python 3.8 + PyTorch 1.13 + Ultralytics 8.2.
2. Exportar modelo a TensorRT FP16:
   ```bash
   yolo export model=best.pt format=engine imgsz=640 half=True
   ```
3. Inference con `stream=True`, `max_det=100` para ~10-12 FPS a 640x640.

## 🔧 Hardware

| Componente | Especificación |
|------------|----------------|
| **Cámara** | USB 2.0 (Sonix 0c49) "ELFO LARANJA" – MJPG 1920x1080 @ 30 fps |
| **LiDAR** | SLAMTEC Mapper M1M1 – 2D, 20 m, 8 Hz, TCP JSON 192.168.11.1:1445 |
| **Edge** | Jetson Nano 4GB JetPack 4.6.1 (CUDA 10.2, torch ≤1.13) |
| **Entrenamiento** | Laptop RTX 3050 4GB VRAM – batch 2/4, 100 epochs ~20-30 min/epoch |

## 📝 Licencia

Proyecto interno ClearSky – uso educativo / investigación.

---

## 🔀 Cómo subir a GitHub (para el mantenedor)

```bash
cd /home/santiago/Proyectos/python/ClearSkypy
git init
git add .
git commit -m "Initial commit: ClearSky detection + segmentation training & test scripts"
# Crear repo en GitHub y pushear:
# gh repo create ClearSky --public --source=. --push
# o manualmente:
# git remote add origin https://github.com/<usuario>/ClearSky.git
# git push -u origin main
```

**Archivos ignorados (`.gitignore` recomendado):**
```
__pycache__/
*.pyc
.venv/
src/training/train/runs/
src/training/train/mlflow.db
src/training/train/mlruns/
datasets/
*.pt
*.engine
*.log
```

---

**Contacto:** Yair & Sebastián – ClearSky Team.