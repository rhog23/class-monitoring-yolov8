import cv2, argparse
import supervision as sv
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class Monitoring")
    parser.add_argument("--imgsz", default=192, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        results = model(frame, imgsz=args.imgsz)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

        cv2.imshow(
            "class monitoring",
            label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels
            ),
        )

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO("./models/yolov8n.pt", task="detect")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    main()
