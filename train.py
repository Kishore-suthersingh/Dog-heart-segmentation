from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Train model on your dataset
model.train(data="dataset.yaml", epochs=50, imgsz=640)
