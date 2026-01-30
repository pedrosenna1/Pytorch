import cv2
import numpy as np


def draw_box(frame, tlbr, label="", track_id=None, confidence=None, color=(0, 255, 0), thickness=2):
    """
    Desenha uma bounding box no frame.
    
    Args:
        frame: imagem onde desenhar
        tlbr: [x1, y1, x2, y2] coordenadas da box
        label: nome da classe
        track_id: ID do track
        confidence: confiança da detecção (0-1)
        color: cor RGB da box
        thickness: espessura da linha
    """
    x1, y1, x2, y2 = map(int, tlbr)
    
    # Desenha retângulo
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Prepara label
    if track_id is not None and confidence is not None:
        text = f"{label} ID:{track_id} {confidence:.2f}"
    elif track_id is not None:
        text = f"{label} ID:{track_id}"
    elif confidence is not None:
        text = f"{label} {confidence:.2f}"
    else:
        text = label
    
    # Desenha label com fundo
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Fundo do texto
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), 
                  (x1 + text_width, y1), color, -1)
    
    # Texto
    cv2.putText(frame, text, (x1, y1 - baseline - 2), 
                font, font_scale, (255, 255, 255), font_thickness)


def draw_hud(frame, lines, position="top-left", bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """
    Desenha informações HUD no frame.
    
    Args:
        frame: imagem onde desenhar
        lines: lista de strings para exibir
        position: "top-left", "top-right", "bottom-left", "bottom-right"
        bg_color: cor de fundo RGB
        text_color: cor do texto RGB
    """
    if not lines:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    line_height = 25
    padding = 10
    
    # Calcula dimensões do HUD
    max_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_width = max(max_width, text_width)
    
    hud_width = max_width + 2 * padding
    hud_height = len(lines) * line_height + 2 * padding
    
    # Define posição
    h, w = frame.shape[:2]
    if position == "top-left":
        x, y = 10, 10
    elif position == "top-right":
        x, y = w - hud_width - 10, 10
    elif position == "bottom-left":
        x, y = 10, h - hud_height - 10
    else:  # bottom-right
        x, y = w - hud_width - 10, h - hud_height - 10
    
    # Desenha fundo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + hud_width, y + hud_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Desenha textos
    for i, line in enumerate(lines):
        text_y = y + padding + (i + 1) * line_height - 5
        cv2.putText(frame, line, (x + padding, text_y), 
                    font, font_scale, text_color, font_thickness)
