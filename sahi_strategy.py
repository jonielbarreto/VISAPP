import numpy as np
import supervision as sv
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

class SahiStrategy:
    """Handles core SAHI model instantiation and standalone sliced inference passes."""
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        print(f"Loading SAHI AutoDetection model wrapper on device: CUDA...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="0",
        )

    def run_sahi_slice(self, frame: np.ndarray, slice_h: int = 480, slice_w: int = 480) -> sv.Detections:
        """Executes standard standalone SAHI sliced predictions followed by global NMS."""
        result = get_sliced_prediction(
            frame, self.model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
        )
        sahi_boxes = [pred.bbox.to_xyxy() for pred in result.object_prediction_list]
        sahi_scores = [pred.score.value for pred in result.object_prediction_list]
        sahi_classes = [pred.category.id for pred in result.object_prediction_list]

        if len(sahi_boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(sahi_boxes, dtype=np.float32),
                confidence=np.array(sahi_scores, dtype=np.float32),
                class_id=np.array(sahi_classes, dtype=int)
            )
            #return detections.with_nms(threshold=0.4)
            return detections
        return sv.Detections.empty()
