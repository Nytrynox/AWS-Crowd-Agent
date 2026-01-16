# Infra deployment (CloudFormation)

This folder contains a minimal CloudFormation template to provision:
- S3 bucket for snapshots
- SNS topic + email subscription for alerts
- IAM role for the controller Lambda
- Skeleton Lambda (you will update code to the real handler)

Deploy (optional):
- Package and deploy using the AWS Console or the AWS CLI.
- After stack completes, confirm the SNS subscription email.

Parameters:
- SnapshotBucketName: globally-unique name
- AlertEmail: your email for SNS subscription

Notes:
- The template uses Bedrock model ID as an env var; ensure Bedrock is enabled in your region/account.
- Replace the inline Lambda with a proper deployment (zip) pointing to `lambda/bedrock_controller.py`.
