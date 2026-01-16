import argparse
import base64
import json
import time

import boto3

# Minimal Kinesis Video Streams consumer using GetImages (periodic snapshots)
# It polls recent images and forwards them to a target SNS topic or invokes a Lambda directly.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stream', required=True, help='Kinesis Video Stream name')
    ap.add_argument('--topic_arn', help='SNS topic to publish events to')
    ap.add_argument('--lambda_name', help='Lambda to invoke directly')
    ap.add_argument('--camera_id', default='CAM01')
    ap.add_argument('--location', default='Gate A')
    ap.add_argument('--interval', type=int, default=2, help='Seconds between polls')
    ap.add_argument('--region', default=None)
    args = ap.parse_args()

    kvs = boto3.client('kinesisvideo', region_name=args.region)
    get_ep = kvs.get_data_endpoint(StreamName=args.stream, APIName='GET_IMAGES')
    ep = get_ep['DataEndpoint']
    kvs_media = boto3.client('kinesis-video-media', endpoint_url=ep, region_name=args.region)

    sns = boto3.client('sns', region_name=args.region) if args.topic_arn else None
    lam = boto3.client('lambda', region_name=args.region) if args.lambda_name else None

    while True:
        try:
            resp = kvs_media.get_images(StreamName=args.stream, ImageSelectorType='SERVER_TIMESTAMP', MaxResults=1)
            images = resp.get('Images', [])
            if not images:
                time.sleep(args.interval)
                continue
            img = images[0]
            img_bytes = img['ImageContent'].read()
            payload = {
                'camera_id': args.camera_id,
                'location': args.location,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'snapshot_b64': base64.b64encode(img_bytes).decode('utf-8'),
                'cv': {}
            }
            if sns:
                sns.publish(TopicArn=args.topic_arn, Message=json.dumps(payload))
            if lam:
                lam.invoke(FunctionName=args.lambda_name, InvocationType='Event', Payload=json.dumps(payload).encode('utf-8'))
        except Exception:
            pass
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
