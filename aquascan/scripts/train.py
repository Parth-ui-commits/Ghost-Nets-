"""Fine-tune YOLOv8n on side-scan sonar data (YOLO format: data.yaml with train/val paths).
Run on a free Kaggle/Colab GPU:  pip install ultralytics && python scripts/train.py data.yaml"""
import sys
from ultralytics import YOLO
m = YOLO("yolov8n.pt")            # small model -> laptop / edge friendly
m.train(data=sys.argv[1], epochs=60, imgsz=640, batch=16)
m.export(format="onnx")           # optional: faster CPU / edge inference
# then copy runs/detect/train/weights/best.pt -> backend/weights/best.pt
