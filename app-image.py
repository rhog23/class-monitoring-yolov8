import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
from ultralytics import YOLO

print(f"[INFO] 🔵 Loading model ...")
model = YOLO("models/yolov8n.pt", task="detect")
print(f"[INFO] 🟢 Model successfully loaded")
fnt = ImageFont.truetype("arial.ttf", 40)


def predict(inp):
    prediction = model(inp)[0]
    draw = ImageDraw.Draw(inp)

    for label, conf, box in zip(
        prediction.boxes.cls, prediction.boxes.conf, prediction.boxes.xyxy
    ):
        draw.rectangle(box.type(torch.int).tolist(), outline=(255, 0, 0), width=2)
        draw.text(
            tuple((int(box[0]), int(box[1]))),
            f"{prediction.names[label.item()]} {conf:.3%}",
            font=fnt,
            fill=(0, 0, 0),
        )

    return inp


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
).launch()
