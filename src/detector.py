import torch
from ultralytics import YOLO
import cv2

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
        return boxes
