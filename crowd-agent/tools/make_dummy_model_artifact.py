import tarfile
from pathlib import Path

# Creates a tiny model.tar.gz with an empty model.pt placeholder.
# SageMaker requires a model_data S3 URI, but our inference.py falls back to yolov8n.pt.

out = Path('model.tar.gz')
model_dir = Path('model')
model_dir.mkdir(exist_ok=True)
(model_dir / 'model.pt').write_bytes(b'')

with tarfile.open(out, 'w:gz') as tar:
    tar.add(model_dir, arcname='.')

print(f"Wrote {out.resolve()} (placeholder artifact)")
