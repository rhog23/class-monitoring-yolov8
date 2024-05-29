import cv2
import tensorflow as tf
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask

model = YOLO("./models/yolov8n_openvino_model", task="detect")

# video_path = (
#     "data\Camera5_MESIN-08_MESIN-08_20240403191448_20240403193058_898192.mp4".replace(
#         "\\", "\/"
#     )
# )

video_path = "data/video-kelvin.mp4"

cap = cv2.VideoCapture(video_path)
clahe_model = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

while cap.isOpened():
    success, frame = cap.read()
    # frame = cv2.resize(
    #     frame, (int((frame.shape[1] * 0.5)), int((frame.shape[0] * 0.5)))
    # )
    # frame = cv2.resize(frame, (480, 288))
    # frame = cv2.resize(frame, (640, 480))
    frame = (
        tf.image.resize(frame, (480, 288), preserve_aspect_ratio=True)
        .numpy()
        .astype(np.uint8)
    )

    # # unsharp_frame = unsharp_mask(img_as_float(frame), radius=5, amount=1)
    # # unsharp_frame = img_as_ubyte(unsharp_frame)
    # frame_b = clahe_model.apply(frame[:, :, 0])
    # frame_g = clahe_model.apply(frame[:, :, 1])
    # frame_r = clahe_model.apply(frame[:, :, 2])

    # frame_clahe = np.stack((frame_b, frame_g, frame_r), axis=2)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lab[:, :, 0] = clahe_model.apply(lab[:, :, 0])
    frame_clahe = cv2.cvtColor(lab, cv2.COLOR_YCrCb2BGR)
    denoised_frame = cv2.bilateralFilter(frame_clahe, 3, 20, 20)
    if success:
        results = model(denoised_frame, classes=[0], vid_stride=100)

        annotated_frame = results[0].plot(font_size=8)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
