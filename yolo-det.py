from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("./models/yolov8n.pt", task="detect")


cap = cv2.VideoCapture(0)
cap.set(3, 192)
cap.set(4, 192)

while True:
    success, frame = cap.read()

    results = model(frame, imgsz=128)

    for result in results:
        box = result.boxes
        coords = box.xyxy
        if len(coords) >= 1:
            x = int(coords[0][0])
            y = int(coords[0][1])
            w = int(coords[0][2])
            h = int(coords[0][3])

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{model.names[int(box.cls[0])]}",
                (x, y),
                1,
                1,
                (0, 0, 0),
                1,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

    cv2.imshow("detector", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
