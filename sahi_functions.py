import os
import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
import supervision as sv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sahi_strategy import SahiStrategy
from temporal_manager import TemporalConsistencyManager

class SahiDetectionPipeline:
    """Unified pipeline orquestrator for maritime drone slicing and validation loops."""
    def __init__(self, model_path: str, mode: str = "temporal_consistency", confidence_threshold: float = 0.5):
        if mode not in ["sahi_only", "temporal_consistency"]:
            raise ValueError("Invalid mode. Choose either 'sahi_only' or 'temporal_consistency'.")
        
        self.mode = mode
        self.class_names = {0: "surface_object", 1: "vessel"}
        
        self.sahi = SahiStrategy(model_path, confidence_threshold)

        self.min_confirmation_frames = 3
        self.max_memory_frames = 0
        self.edge_tolerance_px = 0 
        self.temporal = TemporalConsistencyManager(
            min_confirmation_frames=self.min_confirmation_frames, 
            max_memory_frames=self.max_memory_frames,
            edge_tolerance_px=self.edge_tolerance_px
        )
        
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def reset_temporal_state(self):
        self.temporal.reset_state()

    def process_frame(self, frame: np.ndarray, dx: float = 0, dy: float = 0) -> sv.Detections:
        """Routes frame inference according to operational pipeline strategies."""
        if self.mode == "temporal_consistency":
            raw_sahi = self.sahi.run_sahi_slice(frame, slice_h=480, slice_w=480)
            return self.temporal.update_and_recover(raw_sahi, frame.shape, dx, dy)
        else:
            return self.sahi.run_sahi_slice(frame, slice_h=480, slice_w=480)

    def run_video_visualization(self, video_path: str):
        """Runs real-time visual streaming and computes system throughput metrics."""
        self.reset_temporal_state()
        window_name = f"Engine Display - Mode: {self.mode}"
        total_frames = 0
        total_processing_time = 0
        prev_gray = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error initializing stream source file: {video_path}")
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            img_h, img_w, _ = frame.shape
            start_time = time.perf_counter()

            dx, dy = 0, 0
            if self.mode == "temporal_consistency":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    p0 = np.array([[img_w//2, img_h//2], [img_w//3, img_h//3], [2*img_w//3, 2*img_h//3]], dtype=np.float32).reshape(-1, 1, 2)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, maxLevel=1)
                    if p1 is not None and len(p1) > 0 and st[0] == 1:
                        motions = (p1 - p0)[st == 1]
                        if len(motions) > 0:
                            dx, dy = np.mean(motions, axis=0)
                prev_gray = gray

            detections = self.process_frame(frame, dx, dy)

            if len(detections) > 0:
                labels = [
                    f"#{tid if tid is not None else ''} {self.class_names.get(cid, 'unknown')} {conf:.2f}"
                    for _, _, conf, cid, tid, _ in detections
                ]
                frame = self.box_annotator.annotate(scene=frame, detections=detections)
                frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            frame_duration = time.perf_counter() - start_time
            total_processing_time += frame_duration
            total_frames += 1

            current_fps = 1.0 / frame_duration if frame_duration > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def run_dataset_error_analysis(self, images_dir: str, labels_dir: str, iou_threshold: float = 0.5):
        """Reads an image directory sequentially and executes real-time error analysis."""
        self.reset_temporal_state()
        window_name = f"Dataset Error Analysis Mode - Strategy: {self.mode}"
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files: return

        prev_gray = None
        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None: continue
            img_h, img_w, _ = frame.shape

            dx, dy = 0.0, 0.0
            if self.mode == "temporal_consistency":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    p0 = np.array([[img_w//2, img_h//2], [img_w//3, img_h//3], [2*img_w//3, 2*img_h//3]], dtype=np.float32).reshape(-1, 1, 2)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, maxLevel=1)
                    if p1 is not None and len(p1) > 0 and st[0] == 1:
                        motions = (p1 - p0)[st == 1]
                        if len(motions) > 0: dx, dy = np.mean(motions, axis=0)
                prev_gray = gray

            detections = self.process_frame(frame, dx=dx, dy=dy)
            gt_boxes, gt_classes = [], []
            label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x_c, y_c, bw, bh = map(float, parts)
                            if int(cls) in [0, 1]:
                                x1 = (x_c - bw / 2) * img_w
                                y1 = (y_c - bh / 2) * img_h
                                x2 = (x_c + bw / 2) * img_w
                                y2 = (y_c + bh / 2) * img_h
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_classes.append(int(cls))

            detections_gt = sv.Detections(xyxy=np.array(gt_boxes), class_id=np.array(gt_classes, dtype=int)) if len(gt_boxes) > 0 else sv.Detections.empty()
            tp_boxes, fp_boxes, fn_boxes = [], [], []
            tp_labels, fp_labels, fn_labels = [], [], []
            matched_gt_indices = set()

            if len(detections) > 0:
                for p_box, p_cls, p_conf in zip(detections.xyxy, detections.class_id, detections.confidence):
                    best_iou = 0
                    best_gt_idx = -1
                    if len(detections_gt) > 0:
                        for g_idx, g_box in enumerate(detections_gt.xyxy):
                            ix1 = max(p_box[0], g_box[0])
                            iy1 = max(p_box[1], g_box[1])
                            ix2 = min(p_box[2], g_box[2])
                            iy2 = min(p_box[3], g_box[3])
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            union = (p_box[2]-p_box[0])*(p_box[3]-p_box[1]) + (g_box[2]-g_box[0])*(g_box[3]-g_box[1]) - inter
                            iou = inter / union if union > 0 else 0
                            if iou > best_iou: best_iou = iou; best_gt_idx = g_idx
                    
                    if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices and p_cls == detections_gt.class_id[best_gt_idx]:
                        matched_gt_indices.add(best_gt_idx); tp_boxes.append(p_box)
                        tp_labels.append(f"TP {self.class_names.get(p_cls)} {p_conf:.2f}")
                    else:
                        fp_boxes.append(p_box)
                        fp_labels.append(f"FP {self.class_names.get(p_cls)} {p_conf:.2f}")

            if len(detections_gt) > 0:
                for g_idx, (g_box, g_cls) in enumerate(zip(detections_gt.xyxy, detections_gt.class_id)):
                    if g_idx not in matched_gt_indices: fn_boxes.append(g_box); fn_labels.append(f"FN {self.class_names.get(g_cls)}")

            an_tp = sv.BoxAnnotator(color=sv.Color.GREEN, color_lookup=sv.ColorLookup.INDEX)
            an_fp = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
            an_fn = sv.BoxAnnotator(color=sv.Color.BLUE, color_lookup=sv.ColorLookup.INDEX)
            an_label = sv.LabelAnnotator(text_scale=0.4, text_padding=4, color_lookup=sv.ColorLookup.INDEX)

            if len(tp_boxes) > 0:
                det_tp = sv.Detections(xyxy=np.array(tp_boxes))
                frame = an_tp.annotate(scene=frame, detections=det_tp)
                frame = an_label.annotate(scene=frame, detections=det_tp, labels=tp_labels)
            if len(fp_boxes) > 0:
                det_fp = sv.Detections(xyxy=np.array(fp_boxes))
                frame = an_fp.annotate(scene=frame, detections=det_fp)
                frame = an_label.annotate(scene=frame, detections=det_fp, labels=fp_labels)
            if len(fn_boxes) > 0:
                det_fn = sv.Detections(xyxy=np.array(fn_boxes))
                frame = an_fn.annotate(scene=frame, detections=det_fn)
                frame = an_label.annotate(scene=frame, detections=det_fn, labels=fn_labels)

            cv2.rectangle(frame, (10, 10), (290, 115), (0, 0, 0), -1)
            cv2.putText(frame, f"TRUE POSITIVES (TP): {len(tp_boxes)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"FALSE POSITIVES (FP): {len(fp_boxes)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(frame, f"FALSE NEGATIVES (FN): {len(fn_boxes)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

    def calculate_static_metrics(self, images_dir: str, labels_dir: str, conf_score: float = 0.2, iou_threshold: float = 0.4, output_txt_path: str = None):
        """
        Calculates strict, un-interpolated static Precision, Recall, and F1-Score for operational profiles.
        Optionally replicates all console log prints directly into a local .txt file.
        """
        import sys
        import os
        import cv2
        import numpy as np

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
            image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if not image_files:
                print(f"Error: No valid images found in target directory: {images_dir}")
                return

            total_tp, total_fp, total_fn = 0, 0, 0
            class_metrics = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}

            prev_gray = None
            for img_name in image_files:
                img_path = os.path.join(images_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is None: 
                    continue
                img_h, img_w, _ = frame.shape
                
                dx, dy = 0.0, 0.0
                if self.mode == "temporal_consistency":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        p0 = np.array([[img_w//2, img_h//2], [img_w//3, img_h//3], [2*img_w//3, 2*img_h//3]], dtype=np.float32).reshape(-1, 1, 2)
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, maxLevel=1)
                        if p1 is not None and len(p1) > 0 and st[0] == 1:
                            motions = (p1 - p0)[st == 1]
                            if len(motions) > 0: 
                                dx, dy = np.mean(motions, axis=0)
                    prev_gray = gray

                # Força a inferência utilizando a compensação dinâmica calculada pelo Optical Flow
                detections = self.process_frame(frame, dx=dx, dy=dy) 

                gt_boxes, gt_classes = [], []
                label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls, x_c, y_c, bw, bh = map(float, parts)
                                if int(cls) in [0, 1]:
                                    x1 = (x_c - bw / 2) * img_w
                                    y1 = (y_c - bh / 2) * img_h
                                    x2 = (x_c + bw / 2) * img_w
                                    y2 = (y_c + bh / 2) * img_h
                                    gt_boxes.append([x1, y1, x2, y2])
                                    gt_classes.append(int(cls))

                matched_gt_indices = set()
                if len(detections) > 0 and len(gt_boxes) > 0:
                    for p_box, p_cls in zip(detections.xyxy, detections.class_id):
                        best_iou, best_gt_idx = 0, -1
                        for g_idx, g_box in enumerate(gt_boxes):
                            ix1, iy1 = max(p_box[0], g_box[0]), max(p_box[1], g_box[1])
                            ix2, iy2 = min(p_box[2], g_box[2]), min(p_box[3], g_box[3])
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            union = (p_box[2]-p_box[0])*(p_box[3]-p_box[1]) + (g_box[2]-g_box[0])*(g_box[3]-g_box[1]) - inter
                            iou = inter / union if union > 0 else 0
                            if iou > best_iou: 
                                best_iou = iou
                                best_gt_idx = g_idx
                        
                        if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices and p_cls == gt_classes[best_gt_idx]:
                            matched_gt_indices.add(best_gt_idx)
                            total_tp += 1
                            class_metrics[p_cls]["tp"] += 1
                        else:
                            total_fp += 1
                            class_metrics[p_cls]["fp"] += 1
                else:
                    # Se não houver Ground Truth em frames vazios, todas as detecções viram Falsos Positivos
                    total_fp += len(detections)
                    for p_cls in detections.class_id:
                        class_metrics[p_cls]["fp"] += 1

                for g_idx, g_cls in enumerate(gt_classes):
                    if g_idx not in matched_gt_indices: 
                        total_fn += 1
                        class_metrics[g_cls]["fn"] += 1

            # --- CÁLCULO DAS ESTATÍSTICAS MATEMÁTICAS OPERACIONAIS ---
            global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
            
            prec_cls0 = class_metrics[0]["tp"] / (class_metrics[0]["tp"] + class_metrics[0]["fp"]) if (class_metrics[0]["tp"] + class_metrics[0]["fp"]) > 0 else 0
            rec_cls0  = class_metrics[0]["tp"] / (class_metrics[0]["tp"] + class_metrics[0]["fn"]) if (class_metrics[0]["tp"] + class_metrics[0]["fn"]) > 0 else 0
            f1_cls0   = 2 * (prec_cls0 * rec_cls0) / (prec_cls0 + rec_cls0) if (prec_cls0 + rec_cls0) > 0 else 0
            
            prec_cls1 = class_metrics[1]["tp"] / (class_metrics[1]["tp"] + class_metrics[1]["fp"]) if (class_metrics[1]["tp"] + class_metrics[1]["fp"]) > 0 else 0
            rec_cls1  = class_metrics[1]["tp"] / (class_metrics[1]["tp"] + class_metrics[1]["fn"]) if (class_metrics[1]["tp"] + class_metrics[1]["fn"]) > 0 else 0
            f1_cls1   = 2 * (prec_cls1 * rec_cls1) / (prec_cls1 + rec_cls1) if (prec_cls1 + rec_cls1) > 0 else 0

            # --- IMPRESSÃO FORMATADA DO RELATÓRIO ---
            print("\n" + "="*60)
            print(f"   STATIC OPERATIONAL METRICS PROFILE [{self.mode.upper()}]")
            print(f"   Configuration Context - Conf: {conf_score} | IoU: {iou_threshold}")
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
                
    def run_dataset_benchmark(self, images_dir: str, labels_dir: str, output_txt_path: str = None) -> dict:
        """Evaluates model performance across a complete folder and renders COCO metrics.""" 
        class DualOutput(object):
            def __init__(self, terminal_stream, file_stream):
                self.terminal = terminal_stream; self.file = file_stream
            def write(self, message):
                self.terminal.write(message)
                if self.file: self.file.write(message)
            def flush(self):
                self.terminal.flush()
                if self.file: self.file.flush()

        f_txt = open(output_txt_path, "w", encoding="utf-8") if output_txt_path else None
        original_global_stdout = sys.stdout
        sys.stdout = DualOutput(original_global_stdout, f_txt)

        try:
            self.reset_temporal_state()
            print(f"\nInitializing COCO Benchmark Engine under strategy mode: {self.mode}")
            image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not image_files: return {}

            coco_gt_dataset = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "surface_object"}, {"id": 1, "name": "vessel"}]}
            coco_predictions = []
            gt_ann_id, latencies = 1, []

            prev_gray = None
            for img_idx, img_name in enumerate(tqdm(image_files, desc="Processing frames through pipeline", file=original_global_stdout)):
                img_path = os.path.join(images_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is None: continue
                img_h, img_w, _ = frame.shape
                image_id = img_idx
                coco_gt_dataset["images"].append({"id": image_id, "width": img_w, "height": img_h, "file_name": img_name})

                start_time = time.perf_counter()
                dx, dy = 0.0, 0.0
                if self.mode == "temporal_consistency":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        p0 = np.array([[img_w//2, img_h//2], [img_w//3, img_h//3], [2*img_w//3, 2*img_h//3]], dtype=np.float32).reshape(-1, 1, 2)
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, maxLevel=1)
                        if p1 is not None and len(p1) > 0 and st[0] == 1:
                            motions = (p1 - p0)[st == 1]
                            if len(motions) > 0: dx, dy = np.mean(motions, axis=0)
                    prev_gray = gray

                detections = self.process_frame(frame, dx=dx, dy=dy)
                latencies.append(time.perf_counter() - start_time)

                if len(detections) > 0:
                    for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
                        x1, y1, x2, y2 = box
                        coco_predictions.append({"image_id": image_id, "category_id": int(cls), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(score)})

                label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls, x_c, y_c, bw, bh = map(float, parts)
                                if int(cls) in [0, 1]:
                                    w, h = bw * img_w, bh * img_h
                                    coco_gt_dataset["annotations"].append({"id": gt_ann_id, "image_id": image_id, "category_id": int(cls), "bbox": [float((x_c * img_w) - (w / 2)), float((y_c * img_h) - (h / 2)), float(w), float(h)], "area": float(w * h), "iscrowd": 0})
                                    gt_ann_id += 1

            coco_gt = COCO(); coco_gt.dataset = coco_gt_dataset; coco_gt.createIndex()
            if len(coco_predictions) == 0:
                coco_dt = COCO()
                coco_dt.dataset = {"images": list(coco_gt.imgs.values()), "annotations": [], "categories": list(coco_gt.cats.values())}
                coco_dt.createIndex()
            else:
                coco_dt = coco_gt.loadRes(coco_predictions)

            def get_coco_stats_dict(gt, dt, cat_ids=None, verbose=False):
                evaluator = COCOeval(gt, dt, 'bbox')
                if cat_ids is not None: evaluator.params.catIds = cat_ids
                evaluator.params.maxDets = [1, 10, 100]
                current_stdout = sys.stdout
                if not verbose: sys.stdout = open(os.devnull, 'w')
                try:
                    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize() 
                finally:
                    if not verbose: sys.stdout.close()
                    sys.stdout = current_stdout
                s = evaluator.stats
                raw_recall = evaluator.eval['recall']
                mar_50 = float(np.mean(raw_recall[0, :, 0, 2])) if raw_recall is not None and len(raw_recall) > 0 else 0.0
                return {"mAP50_95": s[0], "mAP50": s[1], "mAR100": s[8], "mAR50": mar_50, "mAP_S": s[3], "mAP_M": s[4], "mAP_L": s[5]}
                
            mean_latency_ms = float(np.mean(latencies) * 1000) if len(latencies) > 0 else 0.0
            operational_fps = float(1.0 / np.mean(latencies)) if len(latencies) > 0 else 0.0

            print("\n" + "="*60); print(f"   COCO OUTPUT LOGS [{self.mode.upper()}]"); print("="*60)
            stats_global = get_coco_stats_dict(coco_gt, coco_dt, verbose=True)
            stats_cls0 = get_coco_stats_dict(coco_gt, coco_dt, cat_ids=[0], verbose=False)
            stats_cls1 = get_coco_stats_dict(coco_gt, coco_dt, cat_ids=[1], verbose=False)

            print("\n" + "="*60); print(f"   COMPUTING COCO METRICS PROFILE [{self.mode.upper()}]")
            print(" METRIC          | GLOBAL     | CLASS 0 (Obj) | CLASS 1 (Ves)")
            print(f" mAP @50         | {stats_global['mAP50']:0.4f}     | {stats_cls0['mAP50']:0.4f}        | {stats_cls1['mAP50']:0.4f}")
            print(f" mAP @50:95      | {stats_global['mAP50_95']:0.4f}     | {stats_cls0['mAP50_95']:0.4f}        | {stats_cls1['mAP50_95']:0.4f}")
            print(f" mAR @50         | {stats_global['mAR50']:0.4f}     | {stats_cls0['mAR50']:0.4f}        | {stats_cls1['mAR50']:0.4f}")
            print(f" mAR @100        | {stats_global['mAR100']:0.4f}     | {stats_cls0['mAR100']:0.4f}        | {stats_cls1['mAR100']:0.4f}")
            print("-"*60)
            print(f" Mean Latency    : {mean_latency_ms:.2f} ms\n Operational FPS : {operational_fps:.2f}\n" + "="*60 + "\n")

            return {"global": stats_global, "class_0": stats_cls0, "class_1": stats_cls1, "fps": operational_fps}
        finally:
            sys.stdout = original_global_stdout
            if f_txt: f_txt.close()