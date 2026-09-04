# JETSON_ENVIRONMENT.md — ClearSky

> **Este documento es la fuente de verdad del entorno de despliegue de ClearSky en la Jetson Nano.**
> Ningún cambio de dependencia, imagen base, o versión de librería debe aplicarse sin comprobar
> compatibilidad contra lo documentado aquí. Si algo se actualiza en el hardware real, este archivo
> debe actualizarse en el mismo commit.

Última verificación: **2026-09-03**, con comandos ejecutados directamente en el equipo físico.

---

## Arquitectura general del proyecto

**El entrenamiento del modelo NO ocurre en la Jetson.** El flujo es:

```
Entrenamiento / exportación (PC externa, Python moderno, ultralytics)
        │
        ▼
   best.onnx  (formato portable)
        │
        ▼  (copiado al proyecto ClearSky)
   Jetson Nano: trtexec → best.engine  (optimizado para esta GPU exacta)
        │
        ▼
   Jetson Nano: script de inferencia (Python 3.6 + TensorRT + pycuda + OpenCV)
   + cámara → detecciones en tiempo real
```

La Jetson **solo hace inferencia**. No requiere PyTorch, no requiere el paquete
`ultralytics`, no requiere Python 3.7. Todo el stack pesado de entrenamiento
vive fuera del dispositivo.

---

## 🔵 Confirmado en la Jetson (verificado con comandos reales, no inferido)

| Componente | Versión | Verificado con |
|---|---|---|
| Hardware | Jetson Nano (GPU Maxwell, CUDA capability 5.3) | — |
| L4T | R32.7.1 | `cat /etc/nv_tegra_release` |
| JetPack | 4.6.1 | Derivado de L4T R32.7.1 — confirmado con el anuncio oficial de NVIDIA (componentes: L4T R32.7.1, CUDA 10.2, cuDNN 8.2.1, TensorRT 8.2.1, OpenCV 4.1.1) |
| Ubuntu (host) | 18.04.6 | `lsb_release -a` |
| Arquitectura | aarch64 / ARM64 | `uname -m` |
| CUDA | 10.2.300 | `nvcc --version`, `/usr/local/cuda/version.txt` |
| cuDNN | 8.2.1.32 | `dpkg -l \| grep cudnn` |
| TensorRT | 8.2.1.8 | `dpkg -l \| grep tensorrt` |
| OpenCV (sistema, con CUDA) | 4.1.1-2-gd5a58aa75 | `dpkg -l \| grep libopencv` |
| Docker | 20.10.7 | `docker --version` |
| NVIDIA Container Runtime | activo | `docker info` |
| Python (host, sistema) | 3.6.9 | `python3 --version` — **este es el Python que se usa para inferencia** |
| Python 3.7.5 | disponible en el host, **no se usa en este flujo** | `python3.7 --version` |
| RAM física | 4GB (limitante para cualquier build local pesado) | — |

### Nota sobre el metapaquete `nvidia-jetpack`
`apt-cache policy nvidia-jetpack` muestra `Instalados: (ninguno)` — el metapaquete
nunca se instaló vía apt (típico en imágenes SD flasheadas directamente). Esto
**no invalida** la versión de JetPack: se confirma por la revisión exacta de L4T
(R32.7.1), que mapea 1:1 y sin ambigüedad a JetPack 4.6.1 según la documentación
oficial de NVIDIA. Los candidatos que muestra apt (4.6.1 a 4.6.6) son solo el
menú disponible en el repo `r32.7/main`, no el estado real del sistema.

### ⚠️ Regla dura: no actualizar el sistema base sin control
No ejecutar `apt upgrade`, `apt install nvidia-jetpack`, ni actualizar paquetes
`nvidia-l4t-*` en el host. Eso puede mover el L4T de R32.7.1 a otra revisión
(p. ej. R32.7.2+) y romper el match exacto con las imágenes Docker y wheels
documentados aquí.

---

## 🟢 Pipeline de inferencia (dependencias objetivo de ClearSky)

| Componente | Versión / fuente | Motivo |
|---|---|---|
| Formato de intercambio del modelo | ONNX (`opset=12`, `simplify=True`) | Portable entre la PC de entrenamiento y la Jetson |
| Motor de inferencia | TensorRT 8.2.1.8 (ya instalado en el host) | Máximo rendimiento en este hardware; no requiere PyTorch en el dispositivo |
| Herramienta de conversión ONNX → engine | `trtexec` (incluido en JetPack, típicamente en `/usr/src/tensorrt/bin/trtexec`) | CLI, no depende de versión de Python |
| Bindings Python para inferencia | `tensorrt` (Python 3.6) + `pycuda` | Verificar disponibilidad exacta en el host antes de codificar |
| Captura de cámara | OpenCV 4.1.1 (sistema) — GStreamer/`nvarguscamerasrc` si es CSI, `cv2.VideoCapture` si es USB | Ya instalado, con soporte CUDA |
| Python de ejecución | 3.6.9 (sistema, sin cambios) | Coincide con los bindings TensorRT ya instalados |

### 🟡 Pendiente por determinar / validar en la Jetson física
- [x] Confirmar `import tensorrt` funciona en Python 3.6 del host (`python3 -c "import tensorrt; print(tensorrt.__version__)"`) → **8.2.1.8 ✅**
- [x] Confirmar disponibilidad de `pycuda` (instalado o pendiente de `pip install`) → **falta, instalar con pip ✅**
- [x] Confirmar si TensorRT es visible dentro de un contenedor Docker (`l4t-base:r32.7.1` + `--runtime nvidia`) o si requiere instalación explícita vía apt dentro del Dockerfile — hay reportes contradictorios en foros de NVIDIA sobre este punto específico para JetPack 4.x → **runtime nvidia no disponible en Docker ✅**
- [x] Tipo de cámara conectada (CSI vs USB) — determina el pipeline de captura → **USB (UVC WebCam) ✅**
- [ ] Ruta exacta de `trtexec` en este equipo (típicamente `/usr/src/tensorrt/bin/trtexec`)

---

## Historial de decisiones (por qué se descartó cada alternativa)

1. **Compilar PyTorch 1.10 desde código fuente para Python 3.7.5 dentro de un
   contenedor** — descartado. Aunque es técnicamente viable (~10-15h de build,
   requiere swap adicional y clang en vez de gcc por bugs conocidos de NEON en
   el Cortex-A57), es sobre-ingeniería para un caso de uso que es solo
   inferencia. Se documenta la investigación por si en el futuro el proyecto
   necesita entrenar/reentrenar en el dispositivo.
2. **Usar Python 3.6 + imagen `l4t-pytorch:r32.7.1-pth1.10-py3` + paquete
   `ultralytics`** — descartado. El paquete `ultralytics` (PyPI) nunca ha
   soportado Python 3.6 (mínimo histórico: 3.7). Los wheels oficiales de
   PyTorch de NVIDIA para JetPack 4.x solo existen para Python 3.6. Este
   choque de versiones es irreconciliable sin compilar desde fuente.
3. **Exportar a ONNX y correr con TensorRT puro** — **elegido**. Evita el
   choque de versiones por completo, reutiliza el TensorRT ya instalado y
   validado en el equipo, y es el patrón de despliegue recomendado por NVIDIA
   para edge devices con recursos limitados.

---

## Comandos de referencia usados para construir este documento

```bash
cat /etc/nv_tegra_release
lsb_release -a
uname -m
nvcc --version
cat /usr/local/cuda/version.txt
dpkg -l | grep -E 'cudnn|tensorrt|libopencv'
docker --version
docker info
python3 --version
python3.7 --version
apt-cache policy nvidia-jetpack
```