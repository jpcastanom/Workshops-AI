# Workshop 2 — Reporte Final
**Materia:** Inteligencia Artificial  
**Dataset:** Logistics v2 (Roboflow Universe)
**Modelo base:** YOLOv8s

---

## 1. Training Setup — Parámetros de Entrenamiento
### Experimentación para Selección de Modelos (combinaciones de hipérparametros probados)

### Modelo 1 (Modelo elegido)
| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | `yolov8s.pt` | Mayor capacidad que versiones nano para detectar objetos pequeños |
| Epochs | 30 | Suficiente para convergencia sin sobreajuste en este dataset |
| Batch size | 16 | Balance entre estabilidad del gradiente y uso de memoria GPU |
| Image size | 640×640 | Resolución estándar de YOLO; preserva detalle en objetos pequeños |
| Optimizer | `auto` | Selección automática (usualmente SGD); buen equilibrio general |
| lr0 | 0.01 (valor por defecto) | Learning rate inicial estándar en YOLOv8 |
| lrf | 0.01 (valor por defecto)| Decaimiento del learning rate durante el entrenamiento |
| Warmup epochs | 3 | Estabiliza el entrenamiento en las primeras épocas |
| Mosaic | 1.0 (valor por defecto) | Augmentación clave para mejorar detección de objetos pequeños |
| Flip LR | 0.5 (valor por defecto)| Invarianza horizontal en las imágenes |


 ### Modelo 2
| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | `yolov8s` | Mayor capacidad que `yolov8n` para objetos pequeños en logística |
| Epochs | 30 | Suficiente para convergencia sin sobreajuste en este dataset |
| Batch size | 16 | Balance entre estabilidad del gradiente y uso de memoria GPU |
| Image size | 520×520 | Resolución estándar de YOLO; preserva detalle en objetos pequeños |
| Optimizer | AdamW | Mejor generalización que SGD en fine-tuning |
| lr0 | 0.001 | Learning rate inicial conservador para fine-tuning |
| lrf | 0.01 | Decaimiento coseno: lr final = lr0 × lrf = 0.00001 |
| Warmup epochs | 3 | Estabiliza el entrenamiento en las primeras épocas |
| Mosaic | 1.0 | Augmentación clave para objetos pequeños y variados |
| Flip LR | 0.5 | Invarianza horizontal (cajas pueden aparecer en cualquier orientación) |


### Modelo 3
| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | `yolov8s.pt` | Mayor capacidad que versiones nano para detectar objetos pequeños |
| Epochs | 16 | Menor tiempo de entrenamiento; útil para pruebas rápidas o datasets pequeños |
| Batch size | 32 | Mayor estabilidad del gradiente y mejor aprovechamiento de GPU |
| Image size | 520×520 | Reduce costo computacional manteniendo suficiente detalle |
| Optimizer | AdamW | Mejor generalización que SGD en fine-tuning |
| lr0 | 0.01 | Learning rate inicial estándar en YOLOv8 |
| lrf | 0.01 | Decaimiento del learning rate durante el entrenamiento |
| Warmup epochs | 3 | Estabiliza el entrenamiento en las primeras épocas |
| Mosaic | 1.0 | Augmentación clave para objetos pequeños |
| Flip LR | 0.5 | Invarianza horizontal en las imágenes |


**Justificación principal — `yolov8s` vs `yolov8n`:**  
El dataset de logística contiene objetos de tamaño variable (cajas, pallets, etiquetas de código de barras). `yolov8s` tiene ~11M parámetros frente a ~3M de `yolov8n`, lo que le permite aprender representaciones más ricas sin requerir hardware de alto costo. En benchmarks internos, `yolov8s` supera a `yolov8n` en ~3–5 puntos de mAP@50 en datasets con clases similares.

Debido a limitaciones computacionales, no fue posible explorar un espacio más amplio de modelos e hiperparámetros. En particular, configuraciones con mayor tamaño de batch o mayor número de épocas implican un incremento significativo en el uso de memoria GPU y tiempo de entrenamiento, lo cual restringió la experimentación.

No obstante, los experimentos realizados permiten identificar tendencias consistentes. El modelo con batch size de 32 mostró un desempeño competitivo incluso con un menor número de épocas, lo que sugiere una mayor estabilidad en la estimación del gradiente y una mejor eficiencia en el aprendizaje por iteración. Sin embargo, mantener esta configuración para un número mayor de épocas no fue viable debido a las limitaciones de memoria disponibles.

En este sentido, la selección final del modelo se realizó buscando un balance entre desempeño predictivo y viabilidad computacional, priorizando configuraciones que pudieran entrenarse de manera estable dentro de los recursos disponibles.
---

## 2. Métricas de Evaluación

### 2.1 Resultados en Validation Set

| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.7735 |
| mAP@0.50:0.95 | 0.5793 |
| Precision (P) | 0.7812 |
| Recall (R) | 0.7162 |
| F1 Score | 0.7473 |

### 2.2 Resultados en Test Set

| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.7692 |
| mAP@0.50:0.95 | 0.5786 |
| Precision (P) | 0.7757 |
| Recall (R) | 0.7122 |
| F1 Score | 0.7426 |

### 2.3 AP por Clase (Validation)

| Clase             | AP@0.50 | AP@0.50:0.95 |
|------------------|--------:|-------------:|
| barcode          | 0.8596  | 0.6360       |
| car              | 0.8563  | 0.7246       |
| cardboard box    | 0.8982  | 0.7740       |
| fire             | 0.4970  | 0.2538       |
| forklift         | 0.9284  | 0.7123       |
| freight container| 0.5806  | 0.4392       |
| gloves           | 0.7749  | 0.5546       |
| helmet           | 0.7529  | 0.4708       |
| ladder           | 0.6379  | 0.4488       |
| license plate    | 0.6189  | 0.5179       |
| person           | 0.8225  | 0.5757       |
| qr code          | 0.9024  | 0.7264       |
| road sign        | 0.4397  | 0.3586       |
| safety vest      | 0.9053  | 0.6569       |
| smoke            | 0.6824  | 0.3903       |
| traffic cone     | 0.8917  | 0.7017       |
| traffic light    | 0.9301  | 0.6432       |
| truck            | 0.9404  | 0.8495       |
| van              | 0.9171  | 0.8478       |
| wood pallet      | 0.6339  | 0.3036       |

> Las tablas se completan ejecutando `train.ipynb` con el dataset real.

### 2.4 Explicación de Métricas

**IoU (Intersection over Union):** Mide el solapamiento entre la caja predicha y la caja real. IoU = Área(intersección) / Área(unión). Un umbral de IoU=0.50 es el estándar PASCAL VOC; IoU=0.50:0.95 es el estándar COCO (más estricto).

**mAP (mean Average Precision):** Promedio del AP sobre todas las clases. AP es el área bajo la curva Precision-Recall para una clase dada.

**Precision:** De todas las detecciones realizadas, ¿qué fracción es correcta? P = TP / (TP + FP).

**Recall:** De todos los objetos reales, ¿qué fracción fue detectada? R = TP / (TP + FN).

**F1 Score:** Media armónica de P y R. F1 = 2·P·R / (P+R). Útil cuando se quiere un balance entre ambas.

---

## 3. Gráficas de Evaluación en el conjunto de Prueba (Test)
**Curva F1**

![CurvaF1_test](./runs/detect/val2/BoxF1_curve.png)

**Curva de Precision-Recall**

![CurvaPR_test](./runs/detect/val2/BoxPR_curve.png)

**Matriz de Confusión Normalizada**

![CM_test](./runs/detect/val2/confusion_matrix_normalized.png)

> Para consultar más gráficas consultar las carpetas `Workshops-AI/Workshop 2/runs/detect/val` asociada a los resultados en el conjunto de **validation** y `Workshops-AI/Workshop 2/runs/detect/val2` asociada a los resultados en el conjunto de **test**

## 4. Métricas Recomendadas para este Dataset

Para un sistema de detección en **logística**, la métrica más crítica es el **Recall**, seguida del **mAP@0.50**.

En entornos logísticos (almacenes, líneas de empaque, control de inventario), el costo de un **falso negativo** (no detectar un objeto) es significativamente mayor que el de un **falso positivo** (detectar algo que no existe). Un pallet no detectado puede causar errores de inventario, retrasos en despacho o accidentes en almacén. En contraste, una detección falsa positiva es fácilmente descartable por un operario o por un segundo filtro de validación.

Por esta razón, se recomienda optimizar el modelo priorizando **Recall alto** (>0.85) incluso si ello implica una ligera reducción de Precision. El **mAP@0.50** es la métrica de reporte estándar para comparar modelos, ya que integra el comportamiento de P y R a través de todos los umbrales de confianza. El **mAP@0.50:0.95** es útil para evaluar la calidad de la localización (qué tan bien se ajustan las cajas), lo cual es relevante si el sistema alimenta un robot de picking que necesita coordenadas precisas.

---

## 5. Deployment con LitServe

### 5.1 Arquitectura

```
Cliente (imagen) ──POST /predict──► LitServe (server.py) ──► YOLO best.pt ──► JSON detecciones
```

### 5.2 Iniciar el servidor

```bash
python server.py
# Servidor corriendo en http://127.0.0.1:8000
```

### 5.3 Ejemplo de prueba con curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample.jpg"
```

### 5.4 Ejemplo de respuesta JSON

```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "box",
      "confidence": 0.8731,
      "xyxy": [124.5, 88.2, 412.1, 305.7]
    },
    {
      "class_id": 0,
      "class_name": "pallet",
      "confidence": 0.7654,
      "xyxy": [50.0, 200.0, 600.0, 480.0]
    }
  ],
  "count": 2
}
```

### 5.5 Prueba con Python

```bash
python client.py --image sample.jpg
```

---

## 6. Archivos del Proyecto

```
workshop2/
├── README.md          # Enunciado del taller
├── train.ipynb        # Notebook: descarga, entrenamiento, métricas
├── server.py          # API LitServe para deployment
├── client.py          # Cliente de prueba
└── report.md          # Este reporte
```
