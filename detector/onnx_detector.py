"""
Detector usando ONNX Runtime
Otimizado para Raspberry Pi - mais leve que TFLite
"""

import numpy as np
import onnxruntime as ort


class ONNXDetector:
    def __init__(self, model_path="weights/yolov5nu.onnx", conf=0.25):
        """
        Inicializa o detector ONNX.
        
        Args:
            model_path: Caminho para o modelo .onnx
            conf: Confidence threshold
        """
        self.model_path = model_path
        self.conf = conf
        
        # Carregar modelo ONNX
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Obter informações de entrada/saída
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        self.input_height = int(self.input_shape[2]) if len(self.input_shape) > 2 else 640
        self.input_width = int(self.input_shape[3]) if len(self.input_shape) > 3 else 640
        
        print(f"✅ Modelo ONNX carregado: {model_path}")
        print(f"   Input: {self.input_shape}")
        print(f"   Outputs: {len(self.output_names)}")
    
    def detect(self, frame):
        """
        Detecta objetos no frame.
        
        Args:
            frame: Imagem BGR do OpenCV
            
        Returns:
            Lista de detecções no formato:
            [{'cls': int, 'score': float, 'bbox': [x1, y1, x2, y2]}, ...]
        """
        h, w = frame.shape[:2]
        
        # Preparar input
        input_data = self._preprocess(frame)
        
        # Inferência
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Pós-processamento
        detections = self._postprocess(outputs, h, w)
        
        return detections
    
    def _preprocess(self, frame):
        """
        Pré-processar frame para ONNX.
        Redimensiona, normaliza e converte para float32.
        """
        # Redimensionar (usar cv2 para compatibilidade)
        import cv2
        img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Converter BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalizar para [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpor para formato NCHW (necessário para ONNX)
        img = np.transpose(img, (2, 0, 1))
        
        # Adicionar batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _postprocess(self, outputs, h, w):
        """
        Pós-processar outputs do modelo ONNX.
        
        YOLOv5 output: (1, 25200, 85) ou similar
        Formato: [x, y, w, h, objectness, class_scores...]
        """
        detections = []
        
        try:
            # Output do YOLOv5 ONNX
            output = outputs[0][0]  # (num_predictions, 85)
            
            for pred in output:
                # Primeiros 4: coordenadas
                x_center, y_center, width, height = pred[:4]
                
                # Próximo: objectness score
                objectness = float(pred[4])
                
                if objectness < self.conf:
                    continue
                
                # Resto: class scores
                class_scores = pred[5:]
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id] * objectness)
                
                if score < self.conf:
                    continue
                
                # Converter coordenadas de center to corner
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                detections.append({
                    'cls': cls_id,
                    'score': score,
                    'bbox': [x1, y1, x2, y2]
                })
        
        except Exception as e:
            print(f"⚠️ Erro no pós-processamento ONNX: {e}")
        
        return detections
