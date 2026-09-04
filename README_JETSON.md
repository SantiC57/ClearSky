# ClearSky - Despliegue en Jetson Nano (JetPack 4.6.1)

Inferencia **TensorRT nativa** — sin PyTorch, sin Ultralytics, sin Docker.

> **Modelo**: YOLOv8n detect (3 clases: Cardboard, Organic, Plastic)
> **Hardware**: Jetson Nano 4GB (Maxwell, CUDA 5.3)
> **Software**: JetPack 4.6.1 / L4T R32.7.1 / Ubuntu 18.04

---

## ⚠️ Reglas de oro (NO violar)

| Regla | Por qué |
|-------|---------|
| **NO `apt upgrade`** | Rompe L4T R32.7.1 → TensorRT/OpenCV dejan de funcionar |
| **NO instales PyTorch/Ultralytics** | No hay wheels para Python 3.6 en JetPack 4.x; compilar toma 10-15h |
| **NO uses Docker** | `nvidia` runtime no disponible out-of-the-box en JetPack 4.6 |
| **Engine `.engine` solo sirve en ESTA Jetson** | No es portable entre GPUs ni versiones de TensorRT |

---

## 📋 Requisitos previos (ya vienen en JetPack 4.6.1)

| Componente | Versión | Verificación |
|------------|---------|--------------|
| Python (sistema) | 3.6.9 | `python3.6 --version` |
| TensorRT | 8.2.1.8 | `python3.6 -c "import tensorrt; print(tensorrt.__version__)"` |
| CUDA | 10.2.300 | `nvcc --version` |
| cuDNN | 8.2.1.32 | `dpkg -l \| grep cudnn` |
| OpenCV (con CUDA) | 4.1.1 | `python3.6 -c "import cv2; print(cv2.__version__)"` |

---

## 🚀 Despliegue rápido (copia y pega en la Jetson)

```bash
# 1. Clona la rama con inferencia TensorRT
git clone -b feat/tensorrt-inference https://github.com/SantiC57/ClearSky.git
cd ClearSky

# 2. Instala dependencias del sistema (Python 3.6 + numpy + headers)
sudo apt-get update && sudo apt-get install -y python3.6-dev python3-numpy

# 3. Verifica numpy 3.6
python3.6 -c "import numpy; print(numpy.__version__)"
# Debe salir: 1.13.3 (versión del repo Ubuntu 18.04)

# 4. Configura CUDA paths (agrega a ~/.bashrc para persistir)
export CUDA_INC_DIR=/usr/local/cuda/include
export PATH=/usr/local/cuda/bin:$PATH

# 5. Instala pycuda SIN compilar numpy (usa el del sistema)
python3.6 -m pip install --no-build-isolation pycuda

# 6. Verifica pycuda
python3.6 -c "import pycuda.driver as cuda; cuda.init(); print('pycuda OK')"

# 7. Copia best.onnx a weights/ (desde tu PC de entrenamiento)
# scp best.onnx clearsky@<IP_JETSON>:~/ClearSky/weights/
# Verifica:
ls -lh weights/best.onnx

# 8. Convierte ONNX → TensorRT Engine (OBLIGATORIO en la Jetson)
trtexec --onnx=weights/best.onnx --saveEngine=weights/best.engine --fp16 --workspace=1024

# 9. ¡Inferencia!
# Cámara USB local:
python3.6 infer_trt.py --engine weights/best.engine --source 0 --conf 0.55

# IP Webcam (celular en misma WiFi):
python3.6 infer_trt.py --engine weights/best.engine --source "http://192.168.1.8:8080/video" --conf 0.55
```

---

## 📷 Fuentes de video soportadas (`--source`)

| Tipo | Ejemplo | Notas |
|------|---------|-------|
| Cámara USB (índice) | `0`, `1` | Forza MJPG 1280x720@30 automáticamente |
| IP Webcam (HTTP) | `http://192.168.1.8:8080/video` | Misma red WiFi; app "IP Webcam" en Android |
| Archivo de video | `/ruta/video.mp4` | Para pruebas sin cámara |

---

## ⌨️ Controles en ventana de inferencia

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| Ventana muestra | FPS + conteo Cardboard/Organic/Plastic en tiempo real |

---

## 📊 Rendimiento esperado (Jetson Nano 4GB, FP16)

| Métrica | Valor |
|---------|-------|
| **Throughput** | ~20.7 QPS (48 ms/frame) |
| **Latencia GPU** | ~47.8 ms |
| **RAM** | ~800 MB |
| **GPU Memory** | ~3.7 GB |

> Medido con `trtexec` (batch=1, FP16, workspace=1024 MiB)

---

## 🛠️ Troubleshooting

| Problema | Solución |
|----------|----------|
| `Illegal instruction` al importar numpy/pycuda | Usa `python3.6` explícito; reinstala `python3-numpy` via apt; `pip install --no-build-isolation pycuda` |
| `trtexec: command not found` | `export PATH=/usr/src/tensorrt/bin:$PATH` |
| `cuda.h: No such file` | `export CUDA_INC_DIR=/usr/local/cuda/include` |
| `python3.6: command not found` | `sudo apt-get install python3.6 python3.6-dev` |
| Engine load falla | Regenera en la Jetson: `trtexec --onnx=weights/best.onnx --saveEngine=weights/best.engine --fp16` |
| Cámara no abre | Verifica `ls /dev/video*`; usa `--source 1` si video0 no funciona |
| IP Webcam no conecta | Mismo WiFi; abre `http://<IP>:8080/video` en navegador de la Jetson primero |
| FPS bajo (<10) | Verifica `--fp16` en trtexec; cierra otras apps GPU; usa `sudo jetson_clocks` |

---

## 🔁 Si actualizas el modelo (nuevo entrenamiento)

```bash
# En tu PC (con ultralytics):
yolo export model=best.pt format=onnx opset=12 simplify=True

# Copia a Jetson:
scp best.onnx clearsky@<IP_JETSON>:~/ClearSky/weights/best.onnx

# En Jetson - regenera engine:
cd ~/ClearSky
trtexec --onnx=weights/best.onnx --saveEngine=weights/best.engine --fp16 --workspace=1024

# Prueba:
python3.6 infer_trt.py --engine weights/best.engine --source 0 --conf 0.55
```

---

## 📁 Estructura del repo (rama `feat/tensorrt-inference`)

```
ClearSky/
├── JETSON_ENVIRONMENT.md      # Fuente de verdad: versiones exactas verificadas
├── README_JETSON.md           # Este archivo
├── infer_trt.py               # Inferencia TensorRT pura (Python 3.6)
├── weights/
│   ├── best.onnx              # Exportado en PC (git-tracked)
│   └── best.engine            # Generado EN LA JETSON (gitignore)
├── .gitignore                 # Ignora .engine, runs, venv, datasets, *.pt
```

---

## 🔗 Enlaces útiles

- [JetPack 4.6.1 Release Notes](https://developer.nvidia.com/embedded/jetpack-4_6_1)
- [TensorRT 8.2 Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
- [trtexec Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)
- [YOLOv8 Export to ONNX](https://docs.ultralytics.com/modes/export/#onnx)

---

## 📝 Licencia

Proyecto interno ClearSky – uso educativo / investigación.

**Contacto**: ClearSky Team