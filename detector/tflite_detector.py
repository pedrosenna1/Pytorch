"""
Detector usando TensorFlow Lite + MobileNet
Otimizado para Raspberry Pi e dispositivos com recursos limitados
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


class TFLiteDetector:
    def __init__(self, model_path="weights/yolov5n-int8.tflite", conf=0.25):
        """
        Inicializa o detector TFLite.
        
        Args:
            model_path: Caminho para o modelo .tflite
            conf: Confidence threshold
        """
        self.model_path = model_path
        self.conf = conf
        
        # Carregar modelo TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obter detalhes de entrada/saída
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Tamanho de entrada esperado
        self.input_shape = self.input_details[0]['shape']
        self.input_height, self.input_width = self.input_shape[1], self.input_shape[2]
        
        print(f"✅ Modelo TFLite carregado: {model_path}")
        print(f"   Entrada: {self.input_shape}")
        print(f"   Outputs: {len(self.output_details)}")
    
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
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Pós-processamento
        detections = self._postprocess(frame, h, w)
        
        return detections
    
    def _preprocess(self, frame):
        """
        Pré-processar frame para TFLite.
        
        Redimensiona, normaliza e converte para o formato esperado.
        """
        # Redimensionar
        img = tf.image.resize(frame, (self.input_height, self.input_width))
        
        # Normalizar para [0, 1]
        img = img / 255.0
        
        # Converter para float32 e adicionar batch dimension
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        
        return img.numpy()
    
    def _postprocess(self, frame, h, w):
        """
        Pós-processar outputs do modelo.
        
        Nota: O formato exato depende do modelo TFLite usado.
        Este código assume formato padrão de YOLO.
        """
        detections = []
        
        try:
            # Saídas típicas de YOLO TFLite:
            # boxes: [1, num_detections, 4] (y1, x1, y2, x2 normalizado)
            # classes: [1, num_detections] (class_id)
            # scores: [1, num_detections] (confidence)
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Se tem múltiplos outputs, processa cada um
            if len(self.output_details) >= 3:
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            else:
                # Single output: assume formato [1, num_detections, 6] (x1,y1,x2,y2,score,cls)
                output_data = output_data[0]
                
                for detection in output_data:
                    x1, y1, x2, y2, score, cls_id = detection
                    
                    if score < self.conf:
                        continue
                    
                    # Converter coordenadas normalizadas para pixels
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                    
                    detections.append({
                        'cls': int(cls_id),
                        'score': float(score),
                        'bbox': [x1, y1, x2, y2]
                    })
                
                return detections
            
            # Processar boxes, classes, scores separados
            for box, cls_id, score in zip(boxes, classes, scores):
                if score < self.conf:
                    continue
                
                # Box em formato [y1, x1, y2, x2] normalizado
                y1, x1, y2, x2 = box
                
                # Converter para pixels
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                
                detections.append({
                    'cls': int(cls_id),
                    'score': float(score),
                    'bbox': [x1, y1, x2, y2]
                })
        
        except Exception as e:
            print(f"⚠️ Erro no pós-processamento TFLite: {e}")
        
        return detections


class TFLiteDetectorAdvanced(TFLiteDetector):
    """
    Versão avançada com suporte a Coral TPU.
    """
    
    def __init__(self, model_path="weights/yolov5n-int8.tflite", conf=0.25, use_coral=False):
        """
        Args:
            model_path: Caminho para o modelo .tflite
            conf: Confidence threshold
            use_coral: Se True, tenta usar Google Coral TPU
        """
        self.use_coral = use_coral
        
        if use_coral:
            try:
                from pycoral.adapters import common
                from pycoral.utils.edgetpu import make_interpreter
                
                print("🪨 Usando Google Coral TPU...")
                # Modelo deve ter "_edgetpu" no nome
                self.interpreter = make_interpreter(model_path)
                self.interpreter.allocate_tensors()
            except ImportError:
                print("⚠️ pycoral não instalado, usando CPU TFLite")
                super().__init__(model_path, conf)
                self.use_coral = False
            except Exception as e:
                print(f"⚠️ Coral TPU não disponível: {e}, usando CPU TFLite")
                super().__init__(model_path, conf)
                self.use_coral = False
        else:
            super().__init__(model_path, conf)
