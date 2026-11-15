import os
import cv2
import time
import torch
import numpy as np
import supervision as sv
from pprint import pprint
from ultralytics import YOLO

class YOLOPipeline:
    """
    A unified pipeline for training, running real-time inference, 
    and profiling execution latency of YOLO architectures.
    """
    def __init__(self, model_path: str):
        """
        Initializes the YOLO pipeline with a specified model checkpoint.
        """
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, data_config_path: str, epochs: int = 200, img_size: int = 1024, batch_size: int = 16):
        """
        Executes model training using predefined hyperparameters configured for critical sensing.
        """
        print(f"Target Execution Device context mapped to: {self.device}")
        
        train_results = self.model.train(
            data=data_config_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            val=True,
            patience=20,
            device=self.device,
            workers=4,
            amp=False,
        )
        return train_results

    def run_video_visualization(self, video_path: str, confidence_threshold: float = 0.5, iou_thr: float = 0.5):
        """
        Executes real-time visual streaming, draws bounding boxes using Supervision,
        and prints operational runtime latency and tracking statistics.
        """
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Consistent mapping pattern for maritime targets
        class_names = {0: "surface_object", 1: "vessel"}
        window_name = "YOLO Real-Time Detection Profile"

        total_frames = 0
        total_processing_time = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream target: {video_path}")
            return

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            start_time = time.perf_counter()

            # Run low-overhead prediction pass
            results = self.model(frame, imgsz=640, conf=confidence_threshold, iou=iou_thr, verbose=False)

            for result in results:
                if len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    detections = sv.Detections(
                        xyxy=xyxy,
                        confidence=scores,
                        class_id=classes
                    )

                    labels = [
                        f"{class_names.get(class_id, 'unknown')} {confidence:.2f}"
                        for _, _, confidence, class_id, _, _ in detections
                    ]

                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            frame_duration = time.perf_counter() - start_time
            total_processing_time += frame_duration
            total_frames += 1

            current_fps = 1.0 / frame_duration if frame_duration > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if total_frames > 0:
            avg_latency_ms = (total_processing_time / total_frames) * 1000
            avg_fps = total_frames / total_processing_time
            print("\n" + "="*50)
            print("          INFERENCE PERFORMANCE REPORT")
            print("="*50)
            print(f" Total Processed Frames  : {total_frames}")
            print(f" Avg Latency per Frame   : {avg_latency_ms:.2f} ms")
            print(f" Avg Processing Rate     : {avg_fps:.2f} FPS")
            print("="*50 + "\n")

    def validate(self, data_config_path: str, img_size: int = 640, conf_score: float = 0.5, iou_thr: float = 0.5):
        """
        Validates performance using internal Ultralytics test-split logic.
        """
        val_img_sz = (1080, 1920) if img_size == 1920 else img_size
        metrics = self.model.val(
            data=data_config_path, 
            split='test', 
            imgsz=val_img_sz, 
            conf=conf_score, 
            iou=iou_thr, 
            save_json=True,
            plots=False
        )
        pprint(metrics.results_dict)
        return os.path.join(metrics.save_dir, "predictions.json")    
    
    def run_dataset_error_analysis(self, images_dir: str, labels_dir: str, img_size: int = 640, confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        """
        Reads an image directory sequentially and executes real-time error analysis for native YOLO,
        color-coding True Positives (Green), False Positives (Red), and False Negatives (Blue).
        """
        window_name = f"YOLO Native Error Analysis Mode - Imgsz: {img_size}"
        class_names = {0: "surface_object", 1: "vessel"}
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"Error: No valid images found in target directory: {images_dir}")
            return

        print(f"\n[ANALYSIS MODE] Iterating over dataset frames. Press 'q' to abort visualization.")

        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None: 
                continue
                
            img_h, img_w, _ = frame.shape
            
            val_img_sz = (1080, 1920) if img_size == 1920 else img_size

            results = self.model(frame, imgsz=val_img_sz, conf=confidence_threshold, verbose=False)
            
            p_boxes, p_classes, p_scores = [], [], []
            for result in results:
                if len(result.boxes) > 0:
                    p_boxes = result.boxes.xyxy.cpu().numpy()
                    p_scores = result.boxes.conf.cpu().numpy()
                    p_classes = result.boxes.cls.cpu().numpy().astype(int)

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

            detections_gt = (
                sv.Detections(xyxy=np.array(gt_boxes), class_id=np.array(gt_classes, dtype=int))
                if len(gt_boxes) > 0 else sv.Detections.empty()
            )

            # Intersection over Union
            tp_boxes, fp_boxes, fn_boxes = [], [], []
            tp_labels, fp_labels, fn_labels = [], [], []
            matched_gt_indices = set()

            if len(p_boxes) > 0:
                for box, cls, conf in zip(p_boxes, p_classes, p_scores):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    if len(detections_gt) > 0:
                        for g_idx, g_box in enumerate(detections_gt.xyxy):
                            # Coordenadas da Interseção
                            ix1 = max(box[0], g_box[0])
                            iy1 = max(box[1], g_box[1])
                            ix2 = min(box[2], g_box[2])
                            iy2 = min(box[3], g_box[3])
                            iw = max(0, ix2 - ix1)
                            ih = max(0, iy2 - iy1)
                            inter = iw * ih
                            
                            area_p = (box[2] - box[0]) * (box[3] - box[1])
                            area_g = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                            union = area_p + area_g - inter
                            
                            iou = inter / union if union > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx
                    
                    # True Positive validation (Same class and IoU above the threshold)
                    if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices and cls == detections_gt.class_id[best_gt_idx]:
                        matched_gt_indices.add(best_gt_idx)
                        tp_boxes.append(box)
                        tp_labels.append(f"TP {class_names.get(cls)} {conf:.2f}")
                    else:
                        fp_boxes.append(box)
                        fp_labels.append(f"FP {class_names.get(cls)} {conf:.2f}")

            # False Negatives
            if len(detections_gt) > 0:
                for g_idx, (g_box, g_cls) in enumerate(zip(detections_gt.xyxy, detections_gt.class_id)):
                    if g_idx not in matched_gt_indices:
                        fn_boxes.append(g_box)
                        fn_labels.append(f"FN {class_names.get(g_cls)}")

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
            cv2.putText(frame, f"FRAME: {img_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"TRUE POSITIVES (TP): {len(tp_boxes)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, f"FALSE POSITIVES (FP): {len(fp_boxes)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(frame, f"FALSE NEGATIVES (FN): {len(fn_boxes)}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
