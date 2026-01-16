# Producer

RTSP -> event publisher. Samples frames and publishes messages to SNS with a base64-encoded JPEG (for snapshot), and leaves CV field empty unless you do edge inference.

Usage:
python producer/rtsp_producer.py --rtsp rtsp://USER:PASS@host/stream --topic_arn arn:aws:sns:... --camera_id CAM01 --location "Gate A" --fps 1

Alternatively run without `--topic_arn` to print payloads locally.
