# Crowd Safety Agent (AWS)

An autonomous AWS-hosted agent that ingests live video, runs computer-vision models to estimate crowd density and detect anomalies, reasons with an LLM (Amazon Bedrock) and autonomously issues alerts, snapshots, and reports (via Lambda, SNS, S3, API Gateway).

This repo includes (Python-only):
- SageMaker training + deployment scripts (YOLO-based MVP, optional CSRNet hook)
- Lambda controller that calls Bedrock for decisions and triggers actions
- RTSP/Kinesis producer for frame ingestion
- Optional dashboard/API stub
- Infra as code (CloudFormation-lite) for core resources
- Local test harness (no AWS required)
 - DynamoDB event store and Status API (API Gateway) with presigned snapshot URLs

Quick starts below are pure Python. No notebooks are required.

## Minimal local demo (no AWS)
1) Create a virtualenv and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put a short test video at `sample.mp4` and run the local pipeline:

```bash
python local/run_local_demo.py --video sample.mp4 --fps 1 --camera_id CAM01 --location "Gate A"
```

This will:
- Run YOLOv8n locally to count people and compute a 3x3 density grid
- Run an optical-flow anomaly heuristic
- Call a Bedrock stub decision-maker (rules) and print actions

## AWS setup (core infra)
Use the CloudFormation template in `infra/template.yaml` via the helper script. You need two buckets: one for snapshots (created by the stack) and one pre-existing bucket to hold the Lambda zip.

End-to-end steps:

```bash
# 1) Package Lambda
make lambda-zip

# 2) Upload handler.zip to a pre-existing code bucket (you create it once in S3 console)
aws s3 cp lambda/handler.zip s3://<your-code-bucket>/lambda/handler.zip

# 3) Deploy stack
cd infra
python deploy_stack.py \
	--stack crowd-agent \
	--template template.yaml \
	--snapshot-bucket <snapshot-bucket-name-to-create> \
	--code-bucket <your-code-bucket> \
	--code-key lambda/handler.zip \
	--alert-email you@example.com \
	--region <aws-region>
```

The stack creates: S3 snapshot bucket, SNS topics (alerts, ingest), DynamoDB events table, Controller Lambda, Status API (CORS), and IAM role. Confirm the SNS email subscription.

## Inference-only (pretrained models)
This project works without training by using pretrained YOLOv8n weights for person detection and an optical-flow heuristic for anomalies. You can:

- Run fully in Lambda by packaging dependencies into the zip (no Docker), or
- Deploy a SageMaker endpoint using the provided `inference.py` that loads `yolov8n.pt` and exposes JSON results and set `SAGEMAKER_ENDPOINT_NAME` so Lambda delegates CV.

Training (optional) instructions remain in `sagemaker/README.md` if you later want to fine-tune.

## Producer (RTSP → SNS)
Run the producer to publish sampled frames as JSON messages (with base64 JPEG):

```bash
python producer/rtsp_producer.py --rtsp rtsp://USER:PASS@HOST/stream --topic_arn arn:aws:sns:REGION:ACCT:Topic --camera_id CAM01 --location "Gate A" --fps 1
```

## Lambda controller (Bedrock)
Handler: `lambda/app.py`. Environment variables:
- `ALERT_TOPIC_ARN`: SNS topic ARN
- `SNAPSHOT_BUCKET`: S3 bucket name for snapshots
- `BEDROCK_MODEL_ID`: Model ID (e.g., anthropic.claude-3-5-sonnet-20240620-v1:0)
 - `SAGEMAKER_ENDPOINT_NAME` (optional): When set, Lambda calls SageMaker for CV and keeps the zip small.
 - `EVENTS_TABLE`: DynamoDB table name for logging events (created by infra template)
 - `BEDROCK_AGENT_ID`, `BEDROCK_AGENT_ALIAS_ID` (optional): Use AgentCore for tool-style decisions; falls back to model call if unset.

Package for deployment (example):

```bash
cd lambda
pip install -r requirements.txt -t .
zip -r handler.zip .
```

Upload `handler.zip` to the Lambda function created by the CloudFormation stack.

If you deploy via the helper script, upload it to your code bucket and pass `--code-bucket` and `--code-key` instead; the stack will reference it automatically.

## Repo structure

crowd-agent/
	├─ infra/                # CloudFormation (core resources)
	├─ sagemaker/            # train.py, inference.py, requirements.txt, deploy_endpoint.py
	├─ lambda/               # bedrock_controller.py, requirements.txt
	├─ producer/             # rtsp_producer.py
	├─ local/                # run_local_demo.py, anomaly.py, bedrock_stub.py
	├─ README.md

## Notes
- For real-time ingest, prefer Kinesis Video Streams with a consumer that calls the SageMaker endpoint, then invokes the Lambda controller.
- Replace the Bedrock stub with real Bedrock in Lambda; ensure the account/region has access to your chosen model.
