import torch
import numpy as np
from pathlib import Path


class YOLOv7Detector:
    def __init__(self, weights, conf=0.4, device="cpu", img_size=640):
        self.device = device
        self.conf = conf
        self.img_size = img_size
        
        # Converter para caminho absoluto
        weights = str(Path(weights).resolve())
        
        print(f"🔄 Carregando modelo de: {weights}")
        
        try:
            # Usar torch.hub para carregar o modelo YOLOv7
            # Agora que a pasta utils foi renomeada para helpers, não há conflito
            self.model = torch.hub.load(
                'WongKinYiu/yolov7',
                'custom',
                path_or_model=weights,
                trust_repo=True,
                force_reload=False
            )
            self.model.conf = self.conf
            self.model.to(self.device)
            
            print(f"✅ Modelo YOLOv7 carregado com sucesso!")
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")

    def detect(self, frame):
        """Detecta objetos no frame"""
        # Inferência
        results = self.model(frame)
        
        # Processar detecções
        dets = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= self.conf:
                x1, y1, x2, y2 = map(int, xyxy)
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(conf),
                    "cls": int(cls),
                })
        
        return dets
