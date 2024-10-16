from ultralytics import YOLO

model = YOLO("yolov8n.pt")
classes = model.names  # Obtém os nomes das classes
print(classes)  # Imprime a lista de classes
    