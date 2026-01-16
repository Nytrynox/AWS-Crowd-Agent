import argparse
import base64
import json
import time
from pathlib import Path

import cv2
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from anomaly import motion_spike


def jpeg_b64(frame):
    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf).decode('utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Path to video file')
    ap.add_argument('--fps', type=float, default=1.0, help='Sample FPS for processing')
    ap.add_argument('--camera_id', default='CAM01')
    ap.add_argument('--location', default='Gate A')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    model = None
    if YOLO is not None:
        try:
            model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"[warn] YOLO load failed ({e}); continuing with no detections fallback.")

    prev_gray = None
    interval = 1.0 / args.fps

    last_time = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last_time < interval:
            continue
        last_time = now

        # Detect people
        if model is not None:
            results = model(frame)
            persons = [b for b in results[0].boxes if int(b.cls) == 0]
            count = len(persons)
            boxes = [bb.xyxy.cpu().numpy().tolist()[0] for bb in persons]
        else:
            # Fallback when YOLO isn't available
            persons = []
            count = 0
            boxes = []

        # Density grid 3x3
        h, w = frame.shape[:2]
        grid = [[0 for _ in range(3)] for _ in range(3)]
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            gx = min(2, int(3 * cx / w))
            gy = min(2, int(3 * cy / h))
            grid[gy][gx] += 1

        # Optical flow anomaly heuristic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        anomaly_flag = False
        motion_mag = 0.0
        if prev_gray is not None:
            anomaly_flag, motion_mag = motion_spike(prev_gray, gray, threshold=1.5)
        prev_gray = gray

        cv_out = {
            'people_count': count,
            'grid_counts': grid,
            'anomaly': anomaly_flag,
            'anomaly_confidence': 0.8 if anomaly_flag else 0.1,
            'motion_magnitude': motion_mag
        }

        event = {
            'camera_id': args.camera_id,
            'location': args.location,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'snapshot_b64': jpeg_b64(frame),
            'cv': cv_out
        }

        # Use stub decision locally
        from bedrock_stub import decide
        decision = decide(event['cv'])
        print(json.dumps({'event': event, 'decision': decision})[:1000])

    cap.release()


if __name__ == '__main__':
    main()
