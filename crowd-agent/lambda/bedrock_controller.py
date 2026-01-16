import os
import json
import base64
import boto3
from datetime import datetime

# Environment variables expected:
# ALERT_TOPIC_ARN, SNAPSHOT_BUCKET, BEDROCK_MODEL_ID

sns = boto3.client('sns')
S3 = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')  # Make sure region supports Bedrock

SYSTEM_PROMPT = (
    "You are CrowdGuard, an autonomous safety agent. Input is structured CV output. "
    "Always minimize false positives. Use tools: save_snapshot(camera_id,timestamp), "
    "send_alert(level,message,recipients). Output JSON keys: action (none|save_snapshot|send_alert|multi), "
    "level (INFO|WARN|CRITICAL), message, recipients (emails/phones), also (optional list of extra actions)."
)


def build_prompt(event):
    cv = event.get('cv', {})
    camera_id = event.get('camera_id', 'UNKNOWN')
    location = event.get('location', 'UNKNOWN')
    ts = event.get('timestamp') or datetime.utcnow().isoformat() + 'Z'
    prompt = f"""
System: {SYSTEM_PROMPT}
Input:
camera_id={camera_id}, location="{location}", people_count={cv.get('people_count')}, 
max_grid={max((max(row) for row in cv.get('grid_counts', [[0]])), default=0)}, 
anomaly={cv.get('anomaly', False)} (conf={cv.get('anomaly_confidence', 0.0)}), motion_mean={cv.get('motion_magnitude', 0.0)}
Answer format: JSON with keys: action, level, message, recipients, also(optional)
"""
    return prompt


def call_bedrock(prompt: str):
    model_id = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
    body = {
        "inputText": prompt
    }
    # The exact API differs per model/provider; adapt as needed.
    resp = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body)
    )
    payload = json.loads(resp.get('body').read()) if hasattr(resp.get('body'), 'read') else json.loads(resp.get('body'))
    # Try common fields
    text = payload.get('outputText') or payload.get('results', [{}])[0].get('outputText') or json.dumps(payload)
    return text


def parse_decision(text: str):
    # Extract first JSON object in text
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        obj = json.loads(text[start:end])
        return obj
    except Exception:
        return {"action": "none", "level": "INFO", "message": "Unable to parse LLM output", "recipients": []}


def send_alert(subject: str, message: dict):
    topic = os.environ['ALERT_TOPIC_ARN']
    sns.publish(TopicArn=topic, Subject=subject, Message=json.dumps(message))


def save_snapshot(camera_id: str, ts: str, snapshot_b64: str | None):
    if not snapshot_b64:
        return
    bucket = os.environ['SNAPSHOT_BUCKET']
    key = f"snapshots/{camera_id}/{ts.replace(':', '-')}.jpg"
    img_bytes = base64.b64decode(snapshot_b64)
    S3.put_object(Bucket=bucket, Key=key, Body=img_bytes, ContentType='image/jpeg')
    return f"s3://{bucket}/{key}"


def lambda_handler(event, context):
    # event should contain: camera_id, location, timestamp, cv{...}, optional snapshot_b64
    prompt = build_prompt(event)
    raw = call_bedrock(prompt)
    decision = parse_decision(raw)

    actions = []
    primary = decision.get('action', 'none')
    also = decision.get('also', []) or []
    if isinstance(also, str):
        also = [also]
    actions = [primary] + also

    outputs = []
    for action in actions:
        if action == 'send_alert':
            level = decision.get('level', 'INFO')
            msg = {
                'level': level,
                'message': decision.get('message', ''),
                'camera_id': event.get('camera_id'),
                'location': event.get('location'),
                'timestamp': event.get('timestamp')
            }
            send_alert(subject=f"CrowdGuard {level}", message=msg)
            outputs.append({'action': 'send_alert', 'status': 'ok'})
        elif action == 'save_snapshot':
            uri = save_snapshot(event.get('camera_id', 'UNKNOWN'), event.get('timestamp') or '', event.get('snapshot_b64'))
            outputs.append({'action': 'save_snapshot', 'uri': uri})
        else:
            outputs.append({'action': action, 'status': 'ignored'})

    return {
        'decision': decision,
        'llm_raw': raw,
        'outputs': outputs
    }
