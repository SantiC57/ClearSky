#!/bin/bash
set -e

CONF_THRESHOLD=${CONF_THRESHOLD:-0.55}
IMGSZ=${IMGSZ:-640}
CAMERA_INDEX=${CAMERA_INDEX:-0}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
FPS=${FPS:-30}
SOURCE=${SOURCE:-}

echo "=== ClearSky Inference ==="
echo "Model: /app/weights/best.pt"
echo "Confidence: ${CONF_THRESHOLD}"
echo "Image size: ${IMGSZ}"
echo "Camera: ${CAMERA_INDEX} (${WIDTH}x${HEIGHT}@${FPS}fps)"

if [ -n "${SOURCE}" ]; then
    echo "Processing file: ${SOURCE}"
    python3 test_model.py --source "${SOURCE}" --conf "${CONF_THRESHOLD}" --imgsz "${IMGSZ}"
else
    echo "Starting webcam inference (press 'q' to quit)..."
    python3 test_model.py --conf "${CONF_THRESHOLD}" --imgsz "${IMGSZ}" \
        --camera "${CAMERA_INDEX}" --width "${WIDTH}" --height "${HEIGHT}" --fps "${FPS}"
fi