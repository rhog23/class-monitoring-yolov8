import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from skimage.util import img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means


class VideoProcessor:
    def __init__(
        self,
        source: int | str = 0,
        model_path: str = "models/yolov8n.onnx",
        target_class: int = 0,
        target_fps: int = 10,
        half: bool = False,
        imgsz: int = 640,
    ) -> None:
        if isinstance(source, int):  # Integer source is webcam ID
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)  # String source is video path
        self.imgsz = imgsz
        self.half = half
        self.model = YOLO(model_path, task="detect")
        self.target_class = target_class
        self.target_fps = target_fps
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
        self.base_sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def basic_sharpening(self, frame: cv2.Mat) -> cv2.Mat:
        return cv2.filter2D(frame, -1, self.base_sharpening_kernel)

    def frame_equalization(self, frame: cv2.Mat) -> cv2.Mat:
        # Histogram equalization using CLAHE

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        lab[:, :, 1] = self.clahe.apply(lab[:, :, 1])
        lab[:, :, 2] = self.clahe.apply(lab[:, :, 2])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def preprocess_frame(self, frame: cv2.Mat) -> cv2.Mat:
        # Resize
        frame = cv2.resize(frame, (480, 288))

        # Denoising
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 3, 3)
        frame = cv2.bilateralFilter(frame, 9, 20, 20)

        # CLAHE for color enhancement
        frame_clahe = self.frame_equalization(frame)

        return frame_clahe

    def process(self) -> None:
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if not success:
                break

            preprocessed_frame = self.preprocess_frame(frame)
            results = self.model(
                preprocessed_frame,
                classes=[self.target_class],
                # vid_stride=int(self.cap.get(cv2.CAP_PROP_FPS) // self.target_fps),
                vid_stride=50,
                imgsz=self.imgsz,
                half=self.half,
                iou=0.8,
            )

            annotated_frame = results[0].plot(font_size=6)
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get user input for source (webcam or video path)
    source = input("Enter 0 for webcam or video file path: ")

    processor = VideoProcessor(source, model_path="models/yolov8m_openvino_model")
    processor.process()
