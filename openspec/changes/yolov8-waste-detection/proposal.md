# Propuesta: Entrenamiento YOLOv8 para Detección de Residuos Sólidos Urbanos

## Resumen Ejecutivo

Implementar un pipeline completo de entrenamiento para YOLOv8 que permita detectar y clasificar residuos sólidos urbanos (cartón, vidrio, metal, papel, plástico, basura) usando el dataset "Waste Classification" de Roboflow. El sistema debe ser reproducible, escalable y optimizado para despliegue en Jetson Nano 4GB.

## Contexto

### Problema Actual
- El dataset disponible es de **clasificación** (etiquetas a nivel imagen)
- YOLOv8 requiere formato de **detección** (bounding boxes)
- No existe pipeline de entrenamiento estructurado
- Se necesita un modelo optimizado para edge computing (Jetson Nano)

### Objetivos
1. Crear pipeline de entrenamiento reproducible
2. Convertir dataset de clasificación a detección (generar bounding boxes)
3. Entrenar modelo YOLOv8n optimizado para Jetson Nano
4. Exportar modelo en formato TensorRT para inferencia en edge
5. Documentar proceso completo para reproducibilidad

## Alcance

### Incluido
- Pipeline de entrenamiento con YOLOv8n
- Conversión de dataset clasificación → detección
- Sistema de tracking de experimentos con MLflow
- Exportación a ONNX y TensorRT
- Scripts de evaluación y métricas
- Documentación completa
- Tests unitarios e integración

### Excluido
- Inferencia en tiempo real (se hace en clearsky-lidar)
- Fusión LiDAR + cámara
- API REST para inferencia
- Despliegue en producción

## Requisitos Técnicos

### Hardware
- **Entrenamiento**: PC con GPU (RTX 3050 4GB o superior)
- **Inferencia**: Jetson Nano 4GB con JetPack 4.6.1

### Software
- Python 3.8+ (compatible con Jetson)
- PyTorch 2.0+ con CUDA
- Ultralytics YOLOv8
- MLflow para tracking
- OpenCV para procesamiento de imágenes
- TensorRT para optimización en Jetson

### Dataset
- **Fuente**: Waste Classification (Roboflow)
- **Clases**: cardboard, glass, metal, paper, plastic, trash (6 clases)
- **Formato**: YOLO detection (bounding boxes)
- **Split**: train/valid/test (70/20/10)

## Criterios de Aceptación

1. ✅ Pipeline ejecuta sin errores en PC local
2. ✅ Modelo alcanza mAP@0.5 ≥ 0.85 en validación
3. ✅ Modelo exportado funciona en Jetson Nano
4. ✅ Inferencia en Jetson ≥ 10 FPS a 640x640
5. ✅ Todos los tests pasan (unit + integration)
6. ✅ Documentación completa y reproducible

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Dataset no tiene bounding boxes | Alta | Alto | Generar bounding boxes con modelo pre-entrenado + corrección manual |
| Overfitting en dataset pequeño | Media | Medio | Data augmentation, early stopping, dropout |
| Jetson Nano no soporta última versión YOLOv8 | Alta | Alto | Usar versión compatible (8.0.x), exportar a TensorRT |
| Tiempo de entrenamiento muy largo | Media | Medio | Usar YOLOv8n (nano), batch size optimizado |
| Precision baja en clases minoritarias | Media | Alto | Class weighting, focal loss, oversampling |

## Métricas de Éxito

### Entrenamiento
- mAP@0.5 ≥ 0.85
- mAP@0.5:0.95 ≥ 0.60
- Loss convergencia en < 100 epochs
- Tiempo de entrenamiento < 4 horas (RTX 3050)

### Inferencia (Jetson Nano)
- FPS ≥ 10 a 640x640
- Latencia < 100ms por frame
- Uso de memoria < 3GB
- Precision similar a PC (±5%)

## Entregables

1. **Código fuente**
   - `src/training/yolov8/` - Pipeline de entrenamiento
   - `src/training/data/` - Scripts de conversión de dataset
   - `src/training/export/` - Scripts de exportación
   - `tests/` - Tests unitarios e integración

2. **Modelos**
   - `models/yolov8n-waste-best.pt` - Mejor modelo entrenado
   - `models/yolov8n-waste-best.onnx` - Modelo ONNX
   - `models/yolov8n-waste-best.engine` - Modelo TensorRT (Jetson)

3. **Documentación**
   - `docs/training-guide.md` - Guía de entrenamiento
   - `docs/deployment-guide.md` - Guía de despliegue en Jetson
   - `docs/dataset-conversion.md` - Conversión de dataset

4. **Configuración**
   - `configs/yolov8n-waste.yaml` - Configuración de entrenamiento
   - `configs/jetson-inference.yaml` - Configuración de inferencia

## Cronograma Estimado

| Fase | Duración | Entregables |
|------|----------|-------------|
| 1. Setup y conversión de dataset | 2 días | Dataset en formato YOLO |
| 2. Pipeline de entrenamiento | 3 días | Código de entrenamiento funcional |
| 3. Entrenamiento y tuning | 2 días | Modelo entrenado con métricas objetivo |
| 4. Exportación y optimización | 2 días | Modelos ONNX y TensorRT |
| 5. Testing y documentación | 2 días | Tests pasando, docs completas |
| **Total** | **11 días** | **Sistema completo** |

## Dependencias

- Dataset "Waste Classification" descargado y accesible
- PC con GPU para entrenamiento
- Jetson Nano disponible para testing
- API key de Roboflow (opcional, para dataset privado)

## Próximos Pasos

1. Aprobar propuesta
2. Crear especificaciones detalladas
3. Diseñar arquitectura del sistema
4. Implementar pipeline de entrenamiento
5. Ejecutar entrenamiento y validación
6. Desplegar en Jetson Nano
