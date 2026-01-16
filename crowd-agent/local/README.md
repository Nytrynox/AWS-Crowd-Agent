# Local test harness

- `inference_local.py`: run YOLO locally on an image to get a simple CV JSON.
- `bedrock_stub.py`: a minimal rules engine that mimics Bedrock decisions for offline testing.

Example:
python local/inference_local.py some.jpg | tee cv.json
cat cv.json | python local/bedrock_stub.py
