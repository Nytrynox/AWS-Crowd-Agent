import argparse
import json
import time

import boto3


def deploy_stack(
    stack_name: str,
    template_path: str,
    snapshot_bucket: str,
    code_bucket: str,
    code_key: str,
    alert_email: str,
    region: str | None = None,
    events_table_name: str | None = None,
    ingest_topic_name: str | None = None,
):
    cf = boto3.client('cloudformation', region_name=region)
    with open(template_path, 'r') as f:
        template_body = f.read()

    params = [
        {'ParameterKey': 'SnapshotBucketName', 'ParameterValue': snapshot_bucket},
        {'ParameterKey': 'CodeBucketName', 'ParameterValue': code_bucket},
        {'ParameterKey': 'CodeKey', 'ParameterValue': code_key},
        {'ParameterKey': 'AlertEmail', 'ParameterValue': alert_email},
    ]
    if events_table_name:
        params.append({'ParameterKey': 'EventsTableName', 'ParameterValue': events_table_name})
    if ingest_topic_name:
        params.append({'ParameterKey': 'IngestTopicName', 'ParameterValue': ingest_topic_name})

    # Create or update
    stacks = [s['StackName'] for s in cf.list_stacks(StackStatusFilter=['CREATE_COMPLETE','UPDATE_COMPLETE','UPDATE_ROLLBACK_COMPLETE'])['StackSummaries']]
    if stack_name in stacks:
        print(f"Updating stack {stack_name}...")
        try:
            resp = cf.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=params,
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            waiter = cf.get_waiter('stack_update_complete')
            waiter.wait(StackName=stack_name)
        except cf.exceptions.ClientError as e:
            if 'No updates are to be performed' in str(e):
                print('No updates needed.')
            else:
                raise
    else:
        print(f"Creating stack {stack_name}...")
        resp = cf.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=params,
            Capabilities=['CAPABILITY_NAMED_IAM']
        )
        waiter = cf.get_waiter('stack_create_complete')
        waiter.wait(StackName=stack_name)

    out = cf.describe_stacks(StackName=stack_name)['Stacks'][0]
    print(json.dumps({'StackName': stack_name, 'Outputs': out.get('Outputs', [])}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack', required=True)
    ap.add_argument('--template', default='template.yaml')
    ap.add_argument('--snapshot-bucket', required=True)
    ap.add_argument('--code-bucket', required=True)
    ap.add_argument('--code-key', default='lambda/handler.zip')
    ap.add_argument('--alert-email', required=True)
    ap.add_argument('--region', default=None)
    ap.add_argument('--events-table-name', default=None)
    ap.add_argument('--ingest-topic-name', default=None)
    args = ap.parse_args()
    deploy_stack(
        args.stack,
        args.template,
        args.snapshot_bucket,
        args.code_bucket,
        args.code_key,
        args.alert_email,
        args.region,
        args.events_table_name,
        args.ingest_topic_name,
    )
