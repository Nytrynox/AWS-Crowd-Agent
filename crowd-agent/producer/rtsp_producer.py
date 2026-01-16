import argparse
import base64
import json
import os
import time

import cv2
import boto3

# Simple RTSP frame reader that samples frames and publishes to an SNS topic or API Gateway


def read_frames(rtsp_url, sample_every=15):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP: {rtsp_url}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            yield frame
        idx += 1
    cap.release()


def jpeg_b64(frame):
    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf).decode('utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp', required=True)
    parser.add_argument('--topic_arn', help='SNS topic to publish events to')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--camera_id', default='CAM01')
    parser.add_argument('--location', default='Gate A')
    args = parser.parse_args()

    sns = boto3.client('sns') if args.topic_arn else None

    interval = 1.0 / args.fps
    for frame in read_frames(args.rtsp, sample_every=max(1, int(30/args.fps))):
        payload = {
            'camera_id': args.camera_id,
            'location': args.location,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'snapshot_b64': jpeg_b64(frame),
            'cv': {}  # let downstream add CV if doing producer-only
        }
        if sns:
            sns.publish(TopicArn=args.topic_arn, Message=json.dumps(payload))
        else:
            print(json.dumps({'preview': True, **payload})[:200] + '...')
        time.sleep(interval)


if __name__ == '__main__':
    main()
