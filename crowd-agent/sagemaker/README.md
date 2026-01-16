# SageMaker

Scripts to train and deploy a YOLO-based person detector for MVP.

Files:
- `train.py`: fine-tune YOLO on a YOLO-format dataset in `/opt/ml/input/data/training`
- `inference.py`: inference handler for endpoint (counts persons, 3x3 density grid)
- `requirements.txt`: Python dependencies
- `deploy_endpoint.py`: deploys a model artifact S3 URI as a real-time endpoint

Quick start:
- Package your dataset (YOLO format) and upload to S3.
- Launch training via SageMaker SDK or console, pointing to `train.py`.
- Deploy endpoint with `deploy_endpoint.py`.
