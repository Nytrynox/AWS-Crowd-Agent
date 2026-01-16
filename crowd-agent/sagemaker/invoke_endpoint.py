import argparse
import base64
import json

import boto3
import cv2
import numpy as np


def encode_image(path: str) -> str:
    img = cv2.imread(path)
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--region', default=None)
    args = ap.parse_args()

    runtime = boto3.client('sagemaker-runtime', region_name=args.region)
    body = json.dumps({'image_b64': encode_image(args.image)})
    resp = runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        Body=body,
        ContentType='application/json',
        Accept='application/json'
    )
    out = json.loads(resp['Body'].read())
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
