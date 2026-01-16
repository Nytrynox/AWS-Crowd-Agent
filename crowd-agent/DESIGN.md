# Crowd Safety Agent — Design

High-level architecture and flows based on the provided spec. See README for quick start. This document captures decisions, tradeoffs, and extensions.

- MVP CV: YOLO person detection + 3x3 density grid + optional optical flow heuristic for anomaly.
- Orchestrator: Lambda invokes Bedrock, parses structured JSON response, triggers SNS/S3.
- Ingest: Kinesis Video Streams recommended; for simplicity this repo provides an RTSP producer + SNS. Add a KVS consumer later.
- Storage: S3 for snapshots; DynamoDB optional for event ledger.
- API: API Gateway stub to integrate later for dashboard and status.
- Security: Least privilege IAM, SNS email subscription confirmation required.

Future work:
- Replace heuristic anomaly with SlowFast/I3D or unsupervised AE on SageMaker.
- Multi-model endpoints or MME for cost efficiency.
- Real dashboard with WebSocket + presigned S3 URLs.
- SageMaker Pipelines for retraining.
