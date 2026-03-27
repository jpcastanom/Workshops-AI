import io
from typing import List, Dict, Any

import litserve as ls
from fastapi import UploadFile
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/logistics/y8s/weights/best.pt"


class LogisticsAPI(ls.LitAPI):
    def setup(self, device: str):
        self.model = YOLO(MODEL_PATH)

    def decode_request(self, request: Dict[str, Any]) -> Image.Image:
        if isinstance(request, dict) and "image" in request:
            f: UploadFile = request["image"]
            return Image.open(io.BytesIO(f.file.read())).convert("RGB")
        raise ValueError("Expected multipart form-data with field 'image'")

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model.predict(image, conf=0.25, imgsz=640, verbose=False)[0]
        detections = []
        for b in results.boxes:
            cls_id = int(b.cls[0].item())
            detections.append({
                "class_id": cls_id,
                "class_name": results.names[cls_id],
                "confidence": round(float(b.conf[0].item()), 4),
                "xyxy": [round(float(x), 2) for x in b.xyxy[0].tolist()],
            })
        return detections

    def encode_response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"detections": output, "count": len(output)}


if __name__ == "__main__":
    server = ls.LitServer(LogisticsAPI(), accelerator="auto")
    server.run(port=8000)
