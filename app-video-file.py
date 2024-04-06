import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt", task="detect")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )


sv.process_video(
    source_path="./data/CCTV B.205- 03-04-2024/Camera5_MESIN-08_MESIN-08_20240403191448_20240403193058_898192.mp4",
    target_path="result.mp4",
    callback=callback,
)
