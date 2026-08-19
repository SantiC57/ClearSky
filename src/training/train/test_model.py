"""ClearSky - Prueba del modelo de deteccion con la camara.

Usa el mejor modelo entrenado (YOLO26s detect) en tiempo real con la
camara USB, o sobre una imagen/video si se pasa --source.

Uso:
    python test_model.py                      # camara /dev/video0
    python test_model.py --source video.mp4   # video o imagen
    python test_model.py --conf 0.2           # umbral de confianza
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
# Ruta por defecto al mejor modelo entrenado con YOLOv8n detect (sesión actual).
# Si no existiese, cae al modelo YOLO26s anterior para no romper el flujo.
BEST_MODEL_PATH = Path(
    "/home/santiago/Proyectos/python/ClearSkypy/src/training/train/runs/YOLO_Segmentation/yolo8n_detect_clear_sky/weights/best.pt"
)
if not BEST_MODEL_PATH.exists():
    BEST_MODEL_PATH = CURRENT_DIR / "runs" / "YOLO_Detection" / "yolo26s_detect" / "weights" / "best.pt"

# Clases objetivo (mismo orden que el dataset)
CLASS_NAMES = ["Cardboard", "Organic", "plastic"]


def run_webcam(model: YOLO, conf: float, imgsz: int, camera_index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_index)

    # Forzar MJPG (formato con el que la camara rinde mejor)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {camera_index}")

    print(f"[ClearSky] Camara abierta: {width}x{height} MJPG. Modelo: {Path(BEST_MODEL_PATH).name}")
    print("[ClearSky] Presiona 'q' para salir")

    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ClearSky] No se recibio frame de la camara")
            break

        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        dt = time.perf_counter() - t0

        # Medir FPS suavizado (promedio de los ultimos 15 frames)
        frame_times.append(1.0 / dt if dt > 0 else 0)
        if len(frame_times) > 15:
            frame_times.pop(0)
        fps_actual = sum(frame_times) / len(frame_times)

        annotated = results[0].plot()

        # Overlay de FPS y conteo
        r = results[0]
        counts = {}
        if r.boxes is not None:
            for cls in r.boxes.cls.tolist():
                name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls))
                counts[name] = counts.get(name, 0) + 1
        label = f"FPS: {fps_actual:.1f}  " + "  ".join(f"{k}:{v}" for k, v in counts.items())
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("ClearSky - Deteccion", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_source(model: YOLO, source: str, conf: float, imgsz: int):
    print(f"[ClearSky] Procesando {source} ...")
    results = model.predict(source=source, conf=conf, imgsz=imgsz, save=True)

    for r in results:
        counts = {}
        if r.boxes is not None:
            for cls in r.boxes.cls.tolist():
                name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls))
                counts[name] = counts.get(name, 0) + 1
        print(f"[ClearSky] {r.path}: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Probar el modelo ClearSky con camara o archivo")
    parser.add_argument("--source", type=str, default=None, help="Imagen o video en vez de la camara")
    parser.add_argument("--conf", type=float, default=0.55, help="Umbral de confianza (0.55 recomendado para precisión). Valores menores aumentan recall.")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolucion de inferencia")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de captura de la camara")
    parser.add_argument("--height", type=int, default=720, help="Alto de captura de la camara")
    parser.add_argument("--fps", type=int, default=30, help="FPS de captura de la camara")
    args = parser.parse_args()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {BEST_MODEL_PATH}")

    print(f"[ClearSky] Cargando modelo: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))

    if args.source:
        run_source(model, args.source, args.conf, args.imgsz)
    else:
        run_webcam(model, args.conf, args.imgsz, args.camera, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()