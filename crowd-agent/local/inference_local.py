import base64
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


def run_on_image(path: str):
    img = cv2.imread(path)
    res = model(img)
    people = [b for b in res[0].boxes if int(b.cls) == 0]
    count = len(people)
    boxes = [bb.xyxy.cpu().numpy().tolist()[0] for bb in people]
    return {'people_count': count, 'boxes': boxes}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    args = ap.parse_args()
    out = run_on_image(args.image)
    print(json.dumps(out, indent=2))
