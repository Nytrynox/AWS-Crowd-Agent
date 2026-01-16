import argparse
import json
import time

import boto3
from sagemaker.pytorch import PyTorchModel


parser = argparse.ArgumentParser()
parser.add_argument('--model_s3_uri', required=True)
parser.add_argument('--role_arn', required=True)
parser.add_argument('--instance_type', default='ml.g4dn.xlarge')
parser.add_argument('--endpoint_name', default='crowd-detector-endpoint')
args = parser.parse_args()

sagemaker_session = None  # let SDK pick up from env

model = PyTorchModel(
    model_data=args.model_s3_uri,
    role=args.role_arn,
    entry_point='inference.py',
    source_dir='.',
    framework_version='1.13.1',
    py_version='py39'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=args.instance_type,
    endpoint_name=args.endpoint_name
)

print(json.dumps({'endpoint_name': args.endpoint_name}))
