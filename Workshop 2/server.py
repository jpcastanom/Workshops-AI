import io
from pathlib import Path
from fastapi import UploadFile
from litserve import LitAPI, LitServer
from PIL import Image
from ultralytics import YOLO

# ── Configuración ──────────────────────────────────────────────────────────────

WEIGHTS = Path("/teamspace/studios/this_studio/Workshops-AI/Workshop 2/runs/detect/runs/logistics/y8s/weights/best.pt")

class ObjectDetectionAPI(LitAPI):

    def setup(self, device):
        self.model = YOLO(str(WEIGHTS))
        self.model.to(device)

    def decode_request(self, request: UploadFile) -> Image.Image:
        return Image.open(io.BytesIO(request.file.read())).convert("RGB")

    def predict(self, image: Image.Image):
        return self.model.predict(image, conf=0.25, iou=0.45, imgsz=640,
                                  save=False, verbose=False)

    def encode_response(self, results) -> dict:
        detections = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"detections": detections}

        for cls_id, conf, bbox in zip(
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
            r.boxes.xyxy.cpu().numpy(),
        ):
            detections.append({
                "class_id":   int(cls_id),
                "class_name": r.names[cls_id],
                "confidence": round(float(conf), 4),
                "bbox":       [round(float(x), 2) for x in bbox.tolist()],
            })

        return {"detections": detections}


if __name__ == "__main__":
    server = LitServer(ObjectDetectionAPI(), accelerator="auto")
    server.run(port=8000)