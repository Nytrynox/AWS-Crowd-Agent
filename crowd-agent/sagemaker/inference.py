import base64
import io
import json
import os
from typing import Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

# SageMaker inference script
# Exposes model_fn, input_fn, predict_fn, output_fn as per SageMaker PyTorch convention

_model = None
_prev_gray = None


def model_fn(model_dir: str):
    global _model
    weights = os.path.join(model_dir, 'model.pt')
    if not os.path.exists(weights):
        # fallback to small pretrained model
        weights = 'yolov8n.pt'
    _model = YOLO(weights)
    return _model


def input_fn(request_body: bytes, request_content_type: str):
    if request_content_type == 'application/json':
        payload = json.loads(request_body.decode('utf-8'))
        if 'image_b64' in payload:
            img_bytes = base64.b64decode(payload['image_b64'])
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            return image
        elif 'ndarray' in payload:
            arr = np.array(payload['ndarray'], dtype=np.uint8)
            return arr
    # assume raw bytes of image
    image = cv2.imdecode(np.frombuffer(request_body, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict_fn(input_data, model):
    global _prev_gray
    results = model.predict(source=input_data, verbose=False)
    boxes = results[0].boxes
    people = []
    for b in boxes:
        cls = int(b.cls)
        if cls == 0:  # person
            xyxy = b.xyxy.cpu().numpy().tolist()[0]
            conf = float(b.conf)
            people.append({'bbox': xyxy, 'conf': conf})
    count = len(people)

    # Simple grid density (3x3) based on bbox centers
    h, w = input_data.shape[:2]
    grid = [[0 for _ in range(3)] for _ in range(3)]
    for p in people:
        x1, y1, x2, y2 = p['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx = min(2, int(3 * cx / w))
        gy = min(2, int(3 * cy / h))
        grid[gy][gx] += 1

    # Optical flow anomaly heuristic with state kept across invocations
    anomaly = False
    motion_mag = 0.0
    gray = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
    if _prev_gray is not None and _prev_gray.shape == gray.shape:
        flow = cv2.calcOpticalFlowFarneback(_prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_mag = float(mag.mean())
        anomaly = motion_mag > 1.5
    _prev_gray = gray

    return {
        'people_count': count,
        'grid_counts': grid,
        'boxes': people,
        'image_size': {'w': w, 'h': h},
        'anomaly': anomaly,
        'anomaly_confidence': 0.8 if anomaly else 0.1,
        'motion_magnitude': motion_mag
    }


def output_fn(prediction: Dict[str, Any], accept: str):
    return json.dumps(prediction), 'application/json'
