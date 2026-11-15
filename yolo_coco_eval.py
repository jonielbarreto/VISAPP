import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Dict, Tuple

class COCOEvaluator:
    """
    Handles cross-scaling geometry conversions and coordinates COCO metrics validation
    for tracking small, medium, and large targets uniformly.
    """
    @staticmethod
    def yolo_to_coco_scaled(gt_dir: str, orig_size: tuple, target_size: tuple):
        """
        Converts raw normalized YOLO text annotations into a standardized COCO matrix
        projected onto a generic target scale matrix.
        """
        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}]
        }
        
        scale_w = target_w / orig_w
        scale_height = target_h / orig_h
        annotation_id = 1
        
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
        
        for txt_file in files:
            image_id = os.path.splitext(txt_file)[0]
            coco_data["images"].append({"id": image_id, "width": target_w, "height": target_h})
            
            full_path = os.path.join(gt_dir, txt_file)
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                continue
                
            with open(full_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: 
                        continue
                    
                    cls, x_center, y_center, bbox_w, bbox_h = map(float, parts[:5])
                    
                    # Convert normalized bounding boxes directly to the target canvas scale
                    width = bbox_w * target_w
                    height = bbox_h * target_h
                    x_min = (x_center * target_w) - (width / 2)
                    y_min = (y_center * target_h) - (height / 2)
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(cls) + 1, 
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                    
        return coco_data, scale_w, scale_height

    @classmethod
    def evaluate_predictions(cls, json_pred_file: str, orig_size: tuple, target_size: tuple, gt_dir: str, output_txt_path: str = None):
        """
        Loads inference tracking files, maps bounding boxes down to the resolution scale,
        runs pycocotools verification, prints results to terminal and optionally dumps logs to a .txt file.
        Includes direct matrix extraction for localized mAR@50 metrics.
        """
        class DualOutput(object):
            def __init__(self, terminal_stream, file_stream):
                self.terminal = terminal_stream
                self.file = file_stream
            def write(self, message):
                self.terminal.write(message)
                if self.file:
                    self.file.write(message)
            def flush(self):
                self.terminal.flush()
                if self.file:
                    self.file.flush()

        # Opens the TXT file if the path was provided
        f_txt = open(output_txt_path, "w", encoding="utf-8") if output_txt_path else None
        
        # Redirects the global stdout of the function
        original_global_stdout = sys.stdout
        sys.stdout = DualOutput(original_global_stdout, f_txt)

        try:
            with open(json_pred_file, 'r') as f:
                raw_predictions = json.load(f)

            scale_w = target_size[0] / orig_size[0]
            scale_h = target_size[1] / orig_size[1]

            # Project model output coordinates onto the comparative analysis scale
            for prediction in raw_predictions:
                prediction['image_id'] = str(prediction['image_id'])
                box_x, box_y, box_w, box_h = prediction['bbox']
                
                prediction['bbox'] = [box_x * scale_w, box_y * scale_h, box_w * scale_w, box_h * scale_h]
                prediction['area'] = (box_w * scale_w) * (box_h * scale_h)

            # Generate structural ground truth dictionary projected directly onto target coordinates
            gt_dict, _, _ = cls.yolo_to_coco_scaled(gt_dir, orig_size, target_size)
            
            coco_gt = COCO()
            coco_gt.dataset = gt_dict
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(raw_predictions)

            # Internal function to isolate the evaluation by class
            def get_coco_stats_dict(gt, dt, cat_ids=None, verbose=False):
                evaluator = COCOeval(gt, dt, 'bbox')
                if cat_ids is not None:
                    evaluator.params.catIds = cat_ids
                evaluator.params.maxDets = [1, 10, 100]
                
                current_stdout = sys.stdout
                if not verbose:
                    sys.stdout = open(os.devnull, 'w')
                
                try:
                    evaluator.evaluate()
                    evaluator.accumulate()
                    evaluator.summarize()
                finally:
                    if not verbose:
                        sys.stdout.close()
                    sys.stdout = current_stdout
                
                s = evaluator.stats
                raw_recall = evaluator.eval['recall']
                if raw_recall is not None and len(raw_recall) > 0:
                    mar_50 = float(np.mean(raw_recall[0, :, 0, 2]))
                else:
                    mar_50 = 0.0
                
                return {
                    "mAP50_95": s[0], "mAP50": s[1], 
                    "mAR100": s[8],
                    "mAR50": mar_50,
                    "mAP_S": s[3], "mAP_M": s[4], "mAP_L": s[5]
                }

            print("\n" + "="*60)
            print(f"   COCO OUTPUT LOGS [YOLO:{target_size}]")
            print("="*60)
            stats_global = get_coco_stats_dict(coco_gt, coco_dt, verbose=True)
            
            # Performs isolated extraction by class (IDs 1 and 2 mapped inside GT)
            stats_cls0 = get_coco_stats_dict(coco_gt, coco_dt, cat_ids=[1], verbose=False) 
            stats_cls1 = get_coco_stats_dict(coco_gt, coco_dt, cat_ids=[2], verbose=False)

            print("\n" + "="*60)
            print(f"   COMPUTING COCO METRICS PROFILE [YOLO:{target_size}]")
            print("============================================================")
            print("\n" + "-"*60)
            print(" METRIC          | GLOBAL     | CLASS 0 (Obj) | CLASS 1 (Ves)")
            print("-"*60)
            print(f" mAP @50         | {stats_global['mAP50']:0.4f}     | {stats_cls0['mAP50']:0.4f}        | {stats_cls1['mAP50']:0.4f}")
            print(f" mAP @50:95      | {stats_global['mAP50_95']:0.4f}     | {stats_cls0['mAP50_95']:0.4f}        | {stats_cls1['mAP50_95']:0.4f}")
            print(f" mAR @50         | {stats_global['mAR50']:0.4f}     | {stats_cls0['mAR50']:0.4f}        | {stats_cls1['mAR50']:0.4f}")
            print(f" mAR @100        | {stats_global['mAR100']:0.4f}     | {stats_cls0['mAR100']:0.4f}        | {stats_cls1['mAR100']:0.4f}")
            print("-"*60)
            print(f" mAP Small (S)   | {stats_global['mAP_S']:0.4f}     | {stats_cls0['mAP_S']:0.4f}        | {stats_cls1['mAP_S']:0.4f}")
            print(f" mAP Medium (M)  | {stats_global['mAP_M']:0.4f}     | {stats_cls0['mAP_M']:0.4f}        | {stats_cls1['mAP_M']:0.4f}")
            print(f" mAP Large (L)   | {stats_global['mAP_L']:0.4f}     | {stats_cls0['mAP_L']:0.4f}        | {stats_cls1['mAP_L']:0.4f}")
            print("-"*60 + "\n")

        finally:
            sys.stdout = original_global_stdout
            if f_txt:
                f_txt.close()

    @classmethod
    def calculate_static_metrics(cls, json_pred_file: str, orig_size: tuple, target_size: tuple, gt_dir: str, conf_threshold: float = 0.2, iou_threshold: float = 0.5, output_txt_path: str = None):
        """
        Calculates strict, un-interpolated operational Precision and Recall at a specific 
        confidence and IoU threshold, removing any maxDets ceiling or ranking distortion.
        Optionally replicates all console log prints directly into a local .txt file.
        """
        class DualOutput(object):
            def __init__(self, terminal_stream, file_stream):
                self.terminal = terminal_stream
                self.file = file_stream
            def write(self, message):
                self.terminal.write(message)
                if self.file:
                    self.file.write(message)
            def flush(self):
                self.terminal.flush()
                if self.file:
                    self.file.flush()

        # Abre o buffer do arquivo TXT se o caminho foi fornecido
        f_txt = open(output_txt_path, "w", encoding="utf-8") if output_txt_path else None
        
        # Intercepta e redireciona o stdout global da função
        original_global_stdout = sys.stdout
        sys.stdout = DualOutput(original_global_stdout, f_txt)

        try:
            with open(json_pred_file, 'r') as f:
                raw_predictions = json.load(f)

            # Mapeamento do JSON da YOLO nativa (Inversão: Width, Height)
            target_w, target_h = target_size
            scale_w = target_w / orig_size[0]
            scale_h = target_h / orig_size[1]

            # Agrupa predições por imagem filtrando pelo threshold de confiança estrito
            preds_by_img = {}
            for p in raw_predictions:
                if p.get('score', 1.0) < conf_threshold:
                    continue
                img_id = str(p['image_id'])
                if img_id not in preds_by_img:
                    preds_by_img[img_id] = []
                
                # Reconverte a caixa do JSON do espaço projetado para o espaço do target_size
                bx, by, bw, bh = p['bbox']
                x1 = bx * scale_w
                y1 = by * scale_h
                x2 = (bx + bw) * scale_w
                y2 = (by + bh) * scale_h
                # Ajusta ID de categoria (YOLO salvou como classe + 1 no COCO, voltamos para 0 e 1)
                cat_id = int(p['category_id']) - 1 
                preds_by_img[img_id].append(([x1, y1, x2, y2], cat_id))

            total_tp, total_fp, total_fn = 0, 0, 0
            class_metrics = {
                0: {"tp": 0, "fp": 0, "fn": 0},
                1: {"tp": 0, "fp": 0, "fn": 0}
            }

            # Carrega os arquivos do Ground Truth
            gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.txt')])

            for txt_file in gt_files:
                img_id = os.path.splitext(txt_file)[0]
                
                # Extrai as instâncias reais do arquivo de rótulo (.txt)
                gt_boxes, gt_classes = [], []
                full_gt_path = os.path.join(gt_dir, txt_file)
                if os.path.exists(full_path:=full_gt_path) and os.path.getsize(full_path) > 0:
                    with open(full_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id, x_c, y_c, bw, bh = map(float, parts[:5])
                                if int(cls_id) in [0, 1]:
                                    w = bw * target_w
                                    h = bh * target_h
                                    x1 = (x_c * target_w) - (w / 2)
                                    y1 = (y_c * target_h) - (h / 2)
                                    x2 = (x_c * target_w) + (w / 2)
                                    y2 = (y_c * target_h) + (h / 2)
                                    gt_boxes.append([x1, y1, x2, y2])
                                    gt_classes.append(int(cls_id))

                img_preds = preds_by_img.get(img_id, [])
                matched_gt_indices = set()
                matched_pred_indices = set()

                # Casamento Geométrico Estrito por Interseção sobre União (IoU)
                if len(img_preds) > 0 and len(gt_boxes) > 0:
                    for p_idx, (p_box, p_cls) in enumerate(img_preds):
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for g_idx, g_box in enumerate(gt_boxes):
                            ix1 = max(p_box[0], g_box[0])
                            iy1 = max(p_box[1], g_box[1])
                            ix2 = min(p_box[2], g_box[2])
                            iy2 = min(p_box[3], g_box[3])
                            iw = max(0, ix2 - ix1)
                            ih = max(0, iy2 - iy1)
                            inter = iw * ih
                            
                            area_p = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                            area_g = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                            union = area_p + area_g - inter
                            iou = inter / union if union > 0 else 0
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                        
                        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices and p_cls == gt_classes[best_gt_idx]:
                            matched_gt_indices.add(best_gt_idx)
                            matched_pred_indices.add(p_idx)
                            total_tp += 1
                            class_metrics[p_cls]["tp"] += 1
                        else:
                            total_fp += 1
                            class_metrics[p_cls]["fp"] += 1
                else:
                    # Se não há Ground Truth, todas as predições são Falsos Positivos
                    total_fp += len(img_preds)
                    for _, p_cls in img_preds:
                        class_metrics[p_cls]["fp"] += 1

                # Elementos esquecidos do Ground Truth viram Falsos Negativos
                for g_idx, g_cls in enumerate(gt_classes):
                    if g_idx not in matched_gt_indices:
                        total_fn += 1
                        class_metrics[g_cls]["fn"] += 1

            # Cálculo final da estatística operacional
            global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
            
            prec_cls0 = class_metrics[0]["tp"] / (class_metrics[0]["tp"] + class_metrics[0]["fp"]) if (class_metrics[0]["tp"] + class_metrics[0]["fp"]) > 0 else 0
            rec_cls0 = class_metrics[0]["tp"] / (class_metrics[0]["tp"] + class_metrics[0]["fn"]) if (class_metrics[0]["tp"] + class_metrics[0]["fn"]) > 0 else 0
            f1_cls0 = 2 * (prec_cls0 * rec_cls0) / (prec_cls0 + rec_cls0) if (prec_cls0 + rec_cls0) > 0 else 0
            
            prec_cls1 = class_metrics[1]["tp"] / (class_metrics[1]["tp"] + class_metrics[1]["fp"]) if (class_metrics[1]["tp"] + class_metrics[1]["fp"]) > 0 else 1.0
            rec_cls1 = class_metrics[1]["tp"] / (class_metrics[1]["tp"] + class_metrics[1]["fn"]) if (class_metrics[1]["tp"] + class_metrics[1]["fn"]) > 0 else 0.0
            f1_cls1 = 2 * (prec_cls1 * rec_cls1) / (prec_cls1 + rec_cls1) if (prec_cls1 + rec_cls1) > 0 else 0.0

            print("\n" + "="*60)
            print(f"   STATIC OPERATIONAL METRICS PROFILE [YOLO_NATIVE]")
            print(f"   Configuration Context - Conf: {conf_threshold} | IoU: {iou_threshold}")
            print("============================================================")
            print(" METRIC          | GLOBAL     | CLASS 0 (Obj) | CLASS 1 (Ves)")
            print("-"*60)
            print(f" Static Precision| {global_precision:0.4f}     | {prec_cls0:0.4f}        | {prec_cls1:0.4f}")
            print(f" Static Recall   | {global_recall:0.4f}     | {rec_cls0:0.4f}        | {rec_cls1:0.4f}")
            print(f" Static F1-Score | {global_f1:0.4f}     | {f1_cls0:0.4f}        | {f1_cls1:0.4f}")
            print("="*60 + "\n")
            
            return {
                "precision": global_precision,
                "recall": global_recall,
                "f1_score": global_f1,
                "class_0": {"precision": prec_cls0, "recall": rec_cls0, "f1_score": f1_cls0},
                "class_1": {"precision": prec_cls1, "recall": rec_cls1, "f1_score": f1_cls1}
            }

        finally:
            sys.stdout = original_global_stdout
            if f_txt:
                f_txt.close()