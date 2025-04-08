from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")
model.export(format="onnx")  # Xuất thành "yolov8n-face.onnx"

print("✅ Chuyển đổi YOLOv8 thành công!")
