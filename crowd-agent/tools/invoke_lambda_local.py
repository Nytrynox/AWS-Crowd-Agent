import argparse
import base64
import json
from pathlib import Path

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lambda', dest='lambda_name', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--camera_id', default='CAM01')
    ap.add_argument('--location', default='Gate A')
    args = ap.parse_args()

    lam = boto3.client('lambda')

    img_b64 = base64.b64encode(Path(args.image).read_bytes()).decode('utf-8')
    event = {
        'camera_id': args.camera_id,
        'location': args.location,
        'timestamp': '2025-01-01T00:00:00Z',
        'snapshot_b64': img_b64,
        'cv': {}
    }

    resp = lam.invoke(FunctionName=args.lambda_name, InvocationType='RequestResponse', Payload=json.dumps(event).encode('utf-8'))
    body = resp['Payload'].read()
    print(body.decode('utf-8'))


if __name__ == '__main__':
    main()
