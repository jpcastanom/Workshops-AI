import argparse
import logging
import os

import numpy as np
import requests
import supervision as sv
from PIL import Image

SERVER_URL = "https://127.0.0.1:8000/predict"

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Ruta a la imagen")
    parser.add_argument("--url", default=SERVER_URL, help="URL del endpoint")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        logging.error("Archivo no encontrado: %s", args.image)
        return

    # ── Enviar imagen al servidor ──────────────────────────────────────────────
    with open(args.image, "rb") as f:
        response = requests.post(args.url, files={"request": f}, timeout=60)

    if response.status_code != 200:
        logging.error("Error %s: %s", response.status_code, response.text)
        return

    detections = response.json()["detections"]
    logging.info("%d detecciones recibidas", len(detections))

    if not detections:
        logging.warning("El modelo no detectó ningún objeto.")
        return

    # ── Anotar imagen con supervision ─────────────────────────────────────────
    sv_detections = sv.Detections(
        class_id   = np.array([d["class_id"]   for d in detections]),
        confidence = np.array([d["confidence"] for d in detections]),
        xyxy       = np.array([d["bbox"]       for d in detections]),
    )
    labels = [f"{d['class_name']} {d['confidence']:.2f}" for d in detections]

    annotated = Image.open(args.image)
    annotated = sv.BoxAnnotator().annotate(annotated, sv_detections)
    annotated = sv.LabelAnnotator().annotate(annotated, sv_detections, labels)

    base     = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(os.path.dirname(args.image), f"{base}_annotated.jpg")
    annotated.save(out_path)
    logging.info("Imagen guardada en: %s", out_path)

    # ── Imprimir resumen ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        print(f"  {d['class_name']:<22} conf={d['confidence']:.3f}  "
              f"({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()