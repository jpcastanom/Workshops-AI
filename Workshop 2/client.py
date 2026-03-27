"""
client.py — Prueba el servidor LitServe de detección de objetos.

Uso:
    python client.py --image sample.jpg
    python client.py --image sample.jpg --url http://127.0.0.1:8000/predict
"""
import argparse
import json
import requests
from pathlib import Path


def predict(image_path: str, url: str) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"image": f})
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Ruta a la imagen")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Imagen no encontrada: {args.image}")

    result = predict(args.image, args.url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nTotal detecciones: {result.get('count', len(result.get('detections', [])))}")


if __name__ == "__main__":
    main()
