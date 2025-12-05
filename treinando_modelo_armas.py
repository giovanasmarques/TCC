from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # ou yolov8n.pt (baixe se necessário)
model.train(data='dataset.yaml', epochs=50, imgsz=640, batch=16)
