import tensorflow as tf
from ultralytics import YOLO

# 1. Exportar para SavedModel primeiro
print("1. Exportando para ONNX...")
model = YOLO('weights/yolov5nu.pt')
model.export(format='onnx', imgsz=640)

# 2. Converter ONNX para TFLite usando tf-onnx
print("2. Convertendo ONNX para TFLite...")
import subprocess
subprocess.run(['pip', 'install', 'tf-onnx'], check=True)

import onnx
from tf_onnx import backend

onnx_model = onnx.load('weights/yolov5nu.onnx')
graph_def = backend.onnx_to_tf(onnx_model)

print("3. Salvando como TFLite...")
# Converter para TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([graph_def])
tflite_model = converter.convert()

with open('weights/yolov5nu.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo TFLite gerado com sucesso em 'weights/yolov5nu.tflite'")