#!/usr/bin/env python3
"""
ClearSky - Inferencia TensorRT nativa en Jetson Nano (sin PyTorch/Ultralytics)

Requisitos en Jetson (JetPack 4.6.1):
- Python 3.6 (sistema)
- TensorRT 8.2.1 (ya instalado: import tensorrt)
- pycuda (pip install pycuda)
- OpenCV 4.1.1 con CUDA (ya instalado: import cv2)

Uso:
    # Cámara USB local:
    python3 infer_trt.py --engine weights/best.engine --source 0 --conf 0.55

    # IP Webcam (celular) - misma red WiFi:
    python3 infer_trt.py --engine weights/best.engine --source "http://192.168.1.8:8080/video" --conf 0.55
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401 - inicializa CUDA context
import pycuda.driver as cuda
import tensorrt as trt


# Clases del modelo (mismo orden que entrenamiento)
CLASS_NAMES = ["Cardboard", "Organic", "Plastic"]
NUM_CLASSES = len(CLASS_NAMES)

# Colores para bounding boxes (BGR)
CLASS_COLORS = {
    "Cardboard": (0, 165, 255),   # naranja
    "Organic": (0, 255, 0),       # verde
    "Plastic": (255, 0, 0),       # azul
}


def load_engine(engine_path: str, logger: trt.Logger) -> trt.ICudaEngine:
    """Carga el engine TensorRT serializado desde disco."""
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise RuntimeError(f"Falló deserialización del engine: {engine_path}")
    return engine


def allocate_buffers(engine: trt.ICudaEngine, context: trt.IExecutionContext):
    """Aloca memoria host/device para inputs y outputs."""
    # Asumimos 1 input (images) y 1 output (output0) - típico YOLOv8 ONNX export
    input_name = "images"
    output_name = "output0"

    input_idx = engine.get_binding_index(input_name)
    output_idx = engine.get_binding_index(output_name)

    if input_idx == -1 or output_idx == -1:
        # Fallback: listar bindings para debug
        print("[ClearSky] Bindings del engine:")
        for i in range(engine.num_bindings):
            print(f"  [{i}] {engine.get_binding_name(i)}: shape={engine.get_binding_shape(i)} dtype={engine.get_binding_dtype(i)}")
        raise RuntimeError(f"Bindings esperados no encontrados: input='{input_name}', output='{output_name}'")

    # Shapes fijos (export ONNX con dynamic=False)
    input_shape = (1, 3, 640, 640)
    # Output YOLOv8 detect: (1, 4+num_classes, 8400) -> para 3 clases = (1, 7, 8400)
    output_shape = (1, 4 + NUM_CLASSES, 8400)

    # Memoria device
    d_input = cuda.mem_alloc(np.empty(input_shape, dtype=np.float32).nbytes)
    d_output = cuda.mem_alloc(np.empty(output_shape, dtype=np.float32).nbytes)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    return {
        "d_input": d_input,
        "d_output": d_output,
        "bindings": bindings,
        "stream": stream,
        "input_idx": input_idx,
        "output_idx": output_idx,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    """Preprocesamiento: BGR->RGB, resize, normalize, HWC->CHW, batch."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # <-- FIX: BGR a RGB
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return np.ascontiguousarray(img, dtype=np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list:
    """Non-Maximum Suppression simple (numpy)."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def postprocess(output: np.ndarray, conf_thresh: float, imgsz: int, frame_shape: tuple) -> list:
    """
    Postprocesa output TensorRT (1, 7, 8400) -> lista de detecciones.
    Output layout YOLOv8: [batch, 4+num_classes, 8400]
    4 = cx, cy, w, h (en píxeles del input imgsz x imgsz, NO normalizados 0-1)
    """
    # output shape: (1, 7, 8400) -> transpose to (8400, 7)
    preds = output[0].T  # (8400, 7)

    # Extraer boxes y scores
    boxes = preds[:, :4]      # cx, cy, w, h EN PÍXELES DE imgsz (0-640), no normalizado
    cls_scores = preds[:, 4:] # (8400, 3)

    # Clase con mayor score por predicción
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

    # Filtrar por confidence threshold
    mask = cls_confs >= conf_thresh
    boxes = boxes[mask]
    cls_ids = cls_ids[mask]
    cls_confs = cls_confs[mask]

    if len(boxes) == 0:
        return []

    # Escalar desde espacio del modelo (imgsz x imgsz) al frame real
    h, w = frame_shape[:2]
    scale_x = w / imgsz     # escala desde espacio del modelo al frame real
    scale_y = h / imgsz     # (válido porque preprocess usa resize simple, sin letterbox)

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2) * scale_x
    y1 = (cy - bh / 2) * scale_y
    x2 = (cx + bw / 2) * scale_x
    y2 = (cy + bh / 2) * scale_y
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS por clase
    final_boxes = []
    final_scores = []
    final_cls_ids = []

    for cls_id in range(NUM_CLASSES):
        cls_mask = cls_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_conf = cls_confs[cls_mask]
        keep = nms(cls_boxes, cls_conf, iou_thresh=0.45)
        final_boxes.extend(cls_boxes[keep])
        final_scores.extend(cls_conf[keep])
        final_cls_ids.extend([cls_id] * len(keep))

    # Formato final: lista de dicts
    detections = []
    for box, score, cls_id in zip(final_boxes, final_scores, final_cls_ids):
        detections.append({
            "bbox": box.astype(int),      # x1, y1, x2, y2
            "confidence": float(score),
            "class_id": int(cls_id),
            "class_name": CLASS_NAMES[cls_id],
        })
    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Dibuja bounding boxes y etiquetas en el frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def run_inference(engine_path: str, source: str, conf_thresh: float, imgsz: int):
    """Loop principal de inferencia con cámara o stream HTTP (IP Webcam)."""
    logger = trt.Logger(trt.Logger.WARNING)

    print(f"[ClearSky] Cargando engine: {engine_path}")
    engine = load_engine(engine_path, logger)
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine, context)

    # Soporta índice de cámara (int) o URL de stream HTTP (IP Webcam)
    if source.startswith("http"):
        cap = cv2.VideoCapture(source)
        print(f"[ClearSky] Conectando a stream: {source}")
    else:
        camera_idx = int(source)
        cap = cv2.VideoCapture(camera_idx)
        # Cámara USB - forzar MJPG para mejor rendimiento
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[ClearSky] Cámara {camera_idx} abierta.")

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir fuente: {source}")

    print("[ClearSky] Motor TensorRT listo. Presiona 'q' para salir")

    frame_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ClearSky] No se recibió frame de la cámara")
                break

            t0 = time.perf_counter()

            # Preprocess
            input_data = preprocess(frame, imgsz)

            # Copy H2D
            cuda.memcpy_htod_async(buffers["d_input"], input_data, buffers["stream"])

            # Infer
            context.execute_async_v2(buffers["bindings"], buffers["stream"].handle)

            # Copy D2H
            output = np.empty(buffers["output_shape"], dtype=np.float32)
            cuda.memcpy_dtoh_async(output, buffers["d_output"], buffers["stream"])
            buffers["stream"].synchronize()

            dt = time.perf_counter() - t0

            # Postprocess
            detections = postprocess(output, conf_thresh, imgsz, frame.shape)

            # FPS suavizado
            frame_times.append(1.0 / dt if dt > 0 else 0)
            if len(frame_times) > 15:
                frame_times.pop(0)
            fps = sum(frame_times) / len(frame_times)

            # Draw
            annotated = draw_detections(frame, detections)

            # Overlay FPS + conteos
            counts = {}
            for det in detections:
                name = det["class_name"]
                counts[name] = counts.get(name, 0) + 1
            label = f"FPS: {fps:.1f}  " + "  ".join(f"{k}:{v}" for k, v in counts.items())
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("ClearSky - TensorRT", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Cleanup CUDA
        buffers["d_input"].free()
        buffers["d_output"].free()


def main():
    parser = argparse.ArgumentParser(description="ClearSky - Inferencia TensorRT nativa en Jetson Nano")
    parser.add_argument("--engine", type=str, default="weights/best.engine",
                        help="Ruta al engine TensorRT (.engine)")
    parser.add_argument("--source", type=str, default="0",
                        help="Fuente de video: índice de cámara (ej: 0) o URL stream HTTP (ej: http://192.168.1.8:8080/video)")
    parser.add_argument("--conf", type=float, default=0.55, help="Umbral confianza (default 0.55)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución inferencia (default 640)")
    args = parser.parse_args()

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine no encontrado: {engine_path}. "
                                f"Genera con: trtexec --onnx=weights/best.onnx --saveEngine={engine_path} --fp16 --workspace=1024")

    run_inference(str(engine_path), args.source, args.conf, args.imgsz)


if __name__ == "__main__":
    main()