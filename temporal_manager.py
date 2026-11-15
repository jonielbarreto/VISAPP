import numpy as np
import supervision as sv

class TemporalConsistencyManager:
    """Manages tracking states, multi-frame confirmations, and ego-motion FN buffers."""
    def __init__(self, min_confirmation_frames: int = 3, max_memory_frames: int = 0, edge_tolerance_px: int = 0):
        self.min_confirmation_frames = min_confirmation_frames
        self.max_memory_frames = max_memory_frames
        self.edge_tolerance_px = edge_tolerance_px
        self.tracker = sv.ByteTrack()
        #self.tracker = sv.ByteTrack(
        #    track_activation_threshold=0.20,     # Threshold for detections to become tracklets
        #    minimum_matching_threshold=0.30,     # IoU threshold for matching
        #)
        self.reset_state()

    def reset_state(self):
        """Resets the persistent memory dicts used across consecutive frame sequences."""
        self.state = {
            'history_count': {},
            'bbox_memory': {},
            'frames_missing': {}
        }

    def update_and_recover(self, detections: sv.Detections, img_shape: tuple, dx: float = 0, dy: float = 0) -> sv.Detections:
        """Executes spatial tracking matching and motion-aware memory recovery logic."""
        img_h, img_w = img_shape[:2]
        
        if len(detections) > 0:
            detections = detections.with_nms(threshold=0.2, class_agnostic=True)
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = self.tracker.update_with_detections(sv.Detections.empty())

        final_xyxy, final_ids, final_classes, final_scores = [], [], [], []
        HIGH_CONF_BYPASS = 0.75  

        # Step A: Update active tracklets validated by ByteTrack
        if detections.tracker_id is not None and len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                self.state['history_count'][tracker_id] = self.state['history_count'].get(tracker_id, 0) + 1
                self.state['frames_missing'][tracker_id] = 0
                
                self.state['bbox_memory'][tracker_id] = {
                    'xyxy': detections.xyxy[i],
                    'class_id': detections.class_id[i],
                    'confidence': detections.confidence[i]
                }

                curr_conf = detections.confidence[i]
                count_frames = self.state['history_count'][tracker_id]
                
                if (count_frames >= self.min_confirmation_frames) or (curr_conf >= HIGH_CONF_BYPASS):
                    final_xyxy.append(detections.xyxy[i])
                    final_ids.append(tracker_id)
                    final_classes.append(detections.class_id[i])
                    final_scores.append(curr_conf)

        # Step B: Motion-Compensated False Negative (FN) Buffer Recovery
        current_tracked_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        
        for saved_id in list(self.state['bbox_memory'].keys()):
            if saved_id not in current_tracked_ids:
                data = self.state['bbox_memory'][saved_id]
                x1, y1, x2, y2 = data['xyxy']

                if (x1 <= self.edge_tolerance_px) or (y1 <= self.edge_tolerance_px) or \
                   (x2 >= img_w - self.edge_tolerance_px) or (y2 >= img_h - self.edge_tolerance_px):
                    self.state['bbox_memory'].pop(saved_id, None)
                    self.state['history_count'].pop(saved_id, None)
                    self.state['frames_missing'].pop(saved_id, None)
                    continue

                adjusted_box = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
                self.state['bbox_memory'][saved_id]['xyxy'] = adjusted_box
                self.state['frames_missing'][saved_id] = self.state['frames_missing'].get(saved_id, 0) + 1
                
                if (self.state['frames_missing'][saved_id] <= self.max_memory_frames) and \
                   ((self.state['history_count'].get(saved_id, 0) >= self.min_confirmation_frames) or (data['confidence'] >= HIGH_CONF_BYPASS)):
                    final_xyxy.append(adjusted_box)
                    final_ids.append(saved_id)
                    final_classes.append(data['class_id'])
                    final_scores.append(data['confidence'] * 0.85)
                    
                elif self.state['frames_missing'][saved_id] > self.max_memory_frames:
                    self.state['bbox_memory'].pop(saved_id, None)
                    self.state['history_count'].pop(saved_id, None)
                    self.state['frames_missing'].pop(saved_id, None)

        if len(final_xyxy) > 0:
            return sv.Detections(
                xyxy=np.array(final_xyxy),
                confidence=np.array(final_scores),
                class_id=np.array(final_classes, dtype=int),
                tracker_id=np.array(final_ids, dtype=int)
            )
        return sv.Detections.empty()
