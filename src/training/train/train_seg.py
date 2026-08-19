"""ClearSky - Entrenamiento YOLOv8 segmentation.

Script para entrenar un modelo YOLOv8n-seg sobre el dataset ClearSky
para detectar Cardboard, Organic y plastic con máscaras de segmentación.

Requisitos previos:
- El venv activado: `source .venv/bin/activate.fish`
- El dataset ya organizado en /home/santiago/Proyectos/datasets/Clear_v1.v2-clear_v1.1_-3-tags.yolo26/
- El modelo base yolov8n-seg.pt ya descargado en /home/santiago/Proyectos/python/ClearSkypy/pretrained/

El script usa transfer learning sobre el modelo pre-entrenado en COCO y lo ajusta
a nuestras 3 clases de basura con segmentación (parte visible).
"""

import os
import mlflow
from ultralytics import YOLO

# --- Rutas absolutas ---
PROJECT_ROOT = "/home/santiago/Proyectos/python/ClearSkypy"
# Modelo base oficial de Ultralights YOLOv8n (detección por cajas).
# Usamos .pt de detección porque el dataset actual tiene etiquetas mezcladas
# (algunas boxes, algunas segments). Ultralytics exige consistencia total.
PRETRAINED_MODEL = os.path.join(PROJECT_ROOT, "pretrained", "yolov8n.pt")
DATA_YAML = (
    "/home/santiago/Proyectos/datasets/Clear_v1.v2-clear_v1.1_-3-tags.yolov8/data.yaml"
)
# Carpeta donde se guardarán logs, pesos y gráficas
LOCAL_RUNS_DIR = os.path.join(
    PROJECT_ROOT, "src", "training", "train", "runs", "YOLO_Segmentation"
)

# --- Configuración MLflow ---
MLFLOW_DB = os.path.join(PROJECT_ROOT, "src", "training", "train", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


# --- Configuración de entrenamiento ---
# * batch=2 por defecto: las máscaras consumen más VRAM que las cajas normales.
#   Si al iniciar te sale CUDA out of memory, pon batch=1.
# * epochs=100: mismo esquema que el entrenamiento de detección anterior.
# * imgsz=640: las imágenes originales son 600×450; Ultralytics las escalará con letterbox.
# * patience=20: para early stopping si la validación no mejora.
# * cos_lr=True: decay coseno de learning rate, ya probado y funcionó bien.
def train_segmentation():
    if not os.path.exists(PRETRAINED_MODEL):
        raise FileNotFoundError(
            f"No se encontró el modelo base en {PRETRAINED_MODEL}. "
            "Asegurate de tener yolov8n-seg.pt en la carpeta pretrained/."
        )

    print("🟢 Cargando modelo base yolov8n-seg.pt (transfer learning)...")
    model = YOLO(PRETRAINED_MODEL)  # ultralytics carga y freezea las primeras capas

    print(
        "📦 Iniciando entrenamiento sobre dataset ClearSky (3 clases: Cardboard, Organic, plastic)..."
    )

    # Parámetros elegidos por estabilidad en RTX 3050 (4GB VRAM) y Jetson Nano later.
    model.train(
        data=DATA_YAML,
        epochs=5,
        batch=2,  # Si OOM, cambia a batch=1 y vuelve a correr
        imgsz=640,
        patience=20,
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        workers=4,
        cache=True,  # ~85 MB dataset en RAM → dataloader más rápido
        task="detect",
        project=LOCAL_RUNS_DIR,
        name="yolo8n_detect_clear_sky",
        exist_ok=True,
        save=True,
    )

    print(
        f"🏁 Entrenamiento finalizado. Modelos, logs y gráficas guardados en: {LOCAL_RUNS_DIR}/yolo8n_seg_clear_sky"
    )


if __name__ == "__main__":
    train_segmentation()

