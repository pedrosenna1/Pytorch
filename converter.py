from ultralytics import YOLO

model = YOLO('weights/yolov5n.pt')
model.export(format='tflite', imgsz=640, int8=False)  # Sem INT8 (mais simples)
# Pronto! Gera yolov5n.tflite

print("Modelo TFLite gerado com sucesso em 'weights/yolov5n.tflite'")