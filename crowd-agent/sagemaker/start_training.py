import argparse
import boto3
from sagemaker.pytorch import PyTorch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role-arn', required=True)
    ap.add_argument('--training-s3', required=True, help='S3 URI to training data root (contains dataset.yaml)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--pretrained', default='yolov8n.pt')
    ap.add_argument('--instance-type', default='ml.g4dn.xlarge')
    ap.add_argument('--instance-count', type=int, default=1)
    ap.add_argument('--job-name', default=None)
    args = ap.parse_args()

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=args.role_arn,
        framework_version='1.13.1',
        py_version='py39',
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        hyperparameters={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'imgsz': args.imgsz,
            'pretrained': args.pretrained,
            'dataset_yaml': 'dataset.yaml'
        }
    )

    inputs = {'training': args.training_s3}
    estimator.fit(inputs=inputs, job_name=args.job_name)

    print({'TrainingJobName': estimator.latest_training_job.name, 'ModelArtifacts': estimator.latest_training_job.describe()['ModelArtifacts']['S3ModelArtifacts']})


if __name__ == '__main__':
    main()
