import cv2
import yaml
import numpy as np

from detector.yolov7_detector import YOLOv7Detector
from tracker.bot_sort import BoTSORT
from helpers.counting import UniqueCounter
from helpers.drawing import draw_box, draw_hud


def load_cfg():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_iou(box1, box2):
    """Calcula IoU entre duas boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def filter_overlapping_detections(dets, iou_threshold=0.5):
    """Remove detecções muito sobrepostas da mesma classe, mantendo a de maior confiança"""
    if len(dets) <= 1:
        return dets
    
    # Ordenar por confiança (descendente)
    dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
    
    filtered = []
    for det in dets_sorted:
        # Verificar se overlap com alguma detecção já aceita da mesma classe
        keep = True
        for kept_det in filtered:
            if det[5] == kept_det[5]:  # mesma classe
                iou = calculate_iou(det[:4], kept_det[:4])
                if iou > iou_threshold:
                    keep = False
                    break
        
        if keep:
            filtered.append(det)
    
    return filtered


def main():
    cfg = load_cfg()

    # ===== classes =====
    wanted_classes = cfg["classes"]                  # name -> id
    wanted_ids = set(wanted_classes.values())        # {0,2,3,5,7}
    id_to_name = {v: k for k, v in wanted_classes.items()}

    # ===== detector =====
    detector = YOLOv7Detector(
        weights=cfg["weights"],
        conf=float(cfg["conf_thres"]),
        device=cfg["device"],
    )

    # ===== tracker (BoT-SORT) =====
    tracker = BoTSORT(
        track_high_thresh=float(cfg["track_high_thresh"]),
        track_low_thresh=float(cfg["track_low_thresh"]),
        new_track_thresh=float(cfg["new_track_thresh"]),
        match_thresh=float(cfg["match_thresh"]),
        track_buffer=int(cfg["track_buffer"]),
        with_reid=bool(cfg.get("with_reid", False)),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.5)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.25)),
        gmc_method=cfg.get("gmc_method", "sparseOptFlow"),
        mot20=bool(cfg.get("mot20", False)),
    )

    # ===== counter =====
    counter = UniqueCounter(cfg["multipliers"])
    
    # Dicionário para armazenar confiança original de cada track
    track_confidences = {}

    cap = cv2.VideoCapture(cfg["stream_url"])
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir o stream.")

    print(f"Stream aberto: {cfg['stream_url']}")
    print(f"Classes detectadas: {list(wanted_classes.keys())}")
    print(f"IDs esperados: {wanted_ids}")
    print(f"Multiplicadores: {cfg['multipliers']}")
    print("\nIniciando detecção e tracking...\n")

    frame_count = 0
    import time
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    frame_start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fim do stream ou erro na leitura.")
            break

        frame_count += 1
        fps_frame_count += 1
        
        # Calcular FPS
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        frame_start_time = frame_end_time
        
        # Detecção
        detections = detector.detect(frame)
        
        # DEBUG: Log de todas as detecções antes de filtrar
        if frame_count <= 3:  # Mostrar apenas nos primeiros 3 frames
            print(f"\n[Frame {frame_count}] Detecções brutas do detector: {len(detections)}")
            for d in detections:
                print(f"  cls={d['cls']}, score={d['score']:.2f}, bbox={d['bbox']}")

        # Filtrar apenas classes desejadas e formatar: [x1, y1, x2, y2, score, cls_id]
        # Importante: usar thresholds por classe para reduzir ruído em veículos (evita criar IDs novos toda hora)
        conf_person = float(cfg.get("conf_person", cfg["conf_thres"]))
        conf_vehicle = float(cfg.get("conf_vehicle", max(cfg["conf_thres"], 0.45)))
        conf_bicycle = float(cfg.get("conf_bicycle", max(cfg["conf_thres"], 0.35)))

        dets = []
        det_scores = []  # Salvar scores originais
        for d in detections:
            cls_id = int(d["cls"])
            if cls_id not in wanted_ids:
                continue
            x1, y1, x2, y2 = d["bbox"]
            score = float(d["score"])

            # Threshold por classe
            if cls_id == 0 and score < conf_person:
                continue
            if cls_id == 1 and score < conf_bicycle:
                continue
            if cls_id in (2, 3, 5, 7) and score < conf_vehicle:
                continue

            dets.append([x1, y1, x2, y2, score, cls_id])
            det_scores.append(score)
        
        # Filtrar detecções sobrepostas da mesma classe (reduzir duplicatas)
        dets = filter_overlapping_detections(dets, iou_threshold=0.5)

        dets_np = np.asarray(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

        # Tracking (passar frame_end_time para cálculo de dt)
        tracks = tracker.update(dets_np, frame, frame_time=frame_end_time)

        # Processar tracks
        for t in tracks:
            tid = int(t.track_id)
            tlbr = t.tlbr  # [x1, y1, x2, y2]
            cls_id = int(t.cls_id)
            name = id_to_name.get(cls_id, "unknown")

            # Armazenar confiança original se for novo track
            # Usar score diretamente da array de detecções antes do tracking
            if tid not in track_confidences and len(dets) > 0:
                # Encontrar detecção mais próxima deste track
                best_iou = 0
                best_score = 0.65
                for det in dets:
                    det_bbox = det[:4]
                    iou = calculate_iou(tlbr, det_bbox)
                    if iou > best_iou and det[5] == cls_id:  # mesma classe
                        best_iou = iou
                        best_score = det[4]
                track_confidences[tid] = best_score

            # Contar objeto único
            was_new = tid not in counter.seen_ids.get(name, set())
            counter.observe(name, tid)
            if was_new and frame_count > 1:  # Não logar o primeiro frame
                bbox_area = (tlbr[2]-tlbr[0]) * (tlbr[3]-tlbr[1])
                print(f"[NOVO] {name} ID={tid} pos=({int(tlbr[0])},{int(tlbr[1])}) area={int(bbox_area)} (Total {name}: {counter.raw_counts.get(name, 0)})")

            # Desenhar box
            if cfg.get("draw_boxes", True):
                # Cores diferentes por classe
                colors = {
                    "person": (0, 255, 0),      # Verde
                    "bicycle": (0, 255, 255),   # Amarelo
                    "car": (255, 0, 0),         # Azul
                    "motorcycle": (0, 165, 255), # Laranja
                    "bus": (0, 0, 255),         # Vermelho
                    "truck": (255, 255, 0),     # Ciano
                }
                color = colors.get(name, (255, 255, 255))
                confidence = track_confidences.get(tid, 0.0)
                draw_box(frame, tlbr, label=name, track_id=tid, confidence=confidence, color=color)

        # Preparar HUD
        hud_lines = [
            f"Frame: {frame_count}",
            f"FPS: {current_fps:.1f}",
            f"Deteccoes: {len(dets)}",
            f"Tracks: {len(tracks)}",
            "",
            f"TOTAL Ponderado: {counter.weighted_total():.1f}",
            ""
        ]
        
        for k in sorted(counter.raw_counts.keys()):
            v = counter.raw_counts[k]
            mult = cfg['multipliers'].get(k, 1)
            hud_lines.append(f"{k}: {v} (x{mult} = {v*mult:.1f})")

        draw_hud(frame, hud_lines)

        # Exibir janela
        if cfg.get("show_window", True):
            cv2.imshow("YOLOv7 + BoT-SORT - Contador", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\nEncerrando por comando do usuário...")
                break
            elif key == ord('r'):  # R para resetar contador
                counter = UniqueCounter(cfg["multipliers"])
                print("\nContador resetado!")

        # Log periódico
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | Tracks: {len(tracks)} | Total: {counter.weighted_total():.1f}")

    cap.release()
    if cfg.get("show_window", True):
        cv2.destroyAllWindows()
    
    print("\n=== RESUMO FINAL ===")
    print(f"Total de frames processados: {frame_count}")
    print(f"Total ponderado final: {counter.weighted_total():.1f}")
    print("\nContagem por classe:")
    for k in sorted(counter.raw_counts.keys()):
        v = counter.raw_counts[k]
        mult = cfg['multipliers'].get(k, 1)
        print(f"  {k}: {v} (x{mult} = {v*mult:.1f})")


if __name__ == "__main__":
    main()
