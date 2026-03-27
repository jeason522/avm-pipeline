"""一次性腳本：將 YOLOv8n 匯出為 ONNX 格式"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=11)
print("匯出完成：yolov8n.onnx")
