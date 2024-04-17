import cv2
import numpy as np
from ultralytics import YOLO
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask

model = YOLO("./models/yolov8n.onnx", task="detect")

video_path = "data/20240321213000-20240321215500/Camera14_MESIN-04_MESIN-04_20240321213000_20240321215500_2937239.mp4"

cap = cv2.VideoCapture(video_path)
clahe_model = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))

while cap.isOpened():
    success, frame = cap.read()
    # frame = cv2.resize(
    #     frame, (int((frame.shape[1] * 0.5)), int((frame.shape[0] * 0.5)))
    # )
    frame = cv2.resize(frame, (480, 288))

    # # unsharp_frame = unsharp_mask(img_as_float(frame), radius=5, amount=1)
    # # unsharp_frame = img_as_ubyte(unsharp_frame)
    # frame_b = clahe_model.apply(frame[:, :, 0])
    # frame_g = clahe_model.apply(frame[:, :, 1])
    # frame_r = clahe_model.apply(frame[:, :, 2])

    # frame_clahe = np.stack((frame_b, frame_g, frame_r), axis=2)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab[:, :, 1] = clahe_model.apply(lab[:, :, 1])
    frame_clahe = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
    denoised_frame = cv2.bilateralFilter(frame_clahe, 3, 20, 20)
    if success:
        results = model(denoised_frame, classes=[0], vid_stride=100)

        annotated_frame = results[0].plot(font_size=8)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
