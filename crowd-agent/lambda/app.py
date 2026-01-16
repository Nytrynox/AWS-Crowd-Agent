import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Tuple

import boto3

ALERT_TOPIC_ARN = os.environ.get('ALERT_TOPIC_ARN')
SNAPSHOT_BUCKET = os.environ.get('SNAPSHOT_BUCKET')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
BEDROCK_AGENT_ID = os.environ.get('BEDROCK_AGENT_ID')
BEDROCK_AGENT_ALIAS_ID = os.environ.get('BEDROCK_AGENT_ALIAS_ID')
SAGEMAKER_ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
EVENTS_TABLE = os.environ.get('EVENTS_TABLE')

sns = boto3.client('sns')
s3 = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION) if EVENTS_TABLE else None

_model = None  # lazy load only if running local inference


def decode_image(b64: str):
    # Returns raw JPEG bytes (not decoded) if using SageMaker endpoint
    # or decoded np.ndarray if using local inference
    img_bytes = base64.b64decode(b64)
    if SAGEMAKER_ENDPOINT_NAME:
        return img_bytes
    else:
        import numpy as np
        import cv2
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img


def encode_jpeg(img) -> bytes:
    if SAGEMAKER_ENDPOINT_NAME:
        # If endpoint path used, we never decode; we persist provided bytes
        return img if isinstance(img, (bytes, bytearray)) else bytes(img)
    else:
        import cv2
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            raise RuntimeError('Failed to encode JPEG')
        return buf.tobytes()


def run_cv(img) -> Dict[str, Any]:
    if SAGEMAKER_ENDPOINT_NAME:
        # Invoke SageMaker endpoint with JSON: {image_b64: ...}
        b64 = base64.b64encode(img).decode('utf-8') if isinstance(img, (bytes, bytearray)) else base64.b64encode(bytes(img)).decode('utf-8')
        body = json.dumps({'image_b64': b64})
        resp = sm_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            Body=body,
            ContentType='application/json',
            Accept='application/json'
        )
        return json.loads(resp['Body'].read())
    else:
        # Local inference path using ultralytics in Lambda (requires packaging deps)
        import numpy as np
        import cv2
        from ultralytics import YOLO
        global _model
        if _model is None:
            _model = YOLO('yolov8n.pt')
        res = _model(img)
        people = [b for b in res[0].boxes if int(b.cls) == 0]
        boxes = [bb.xyxy.cpu().numpy().tolist()[0] for bb in people]

        h, w = img.shape[:2]
        grid = [[0 for _ in range(3)] for _ in range(3)]
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            gx = min(2, int(3 * cx / w))
            gy = min(2, int(3 * cy / h))
            grid[gy][gx] += 1

        return {
            'people_count': len(boxes),
            'grid_counts': grid,
            'boxes': boxes,
            'image_size': {'w': w, 'h': h}
        }


def get_prev_snapshot_gray(camera_id: str):
    key = f"snapshots/{camera_id}/latest.jpg"
    try:
        obj = s3.get_object(Bucket=SNAPSHOT_BUCKET, Key=key)
        data = obj['Body'].read()
        if SAGEMAKER_ENDPOINT_NAME:
            # not decoding in Lambda if using SageMaker path
            return data  # raw jpeg for comparison not used here
        else:
            import numpy as np, cv2
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def compute_motion(prev_gray, curr_image) -> Tuple[bool, float]:
    # Motion heuristic computed only in local-inference mode to avoid heavy OpenCV in Lambda zip
    if SAGEMAKER_ENDPOINT_NAME:
        # Motion should be computed in the SageMaker model (enabled in our inference.py)
        return False, 0.0
    else:
        import cv2, numpy as np
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return False, 0.0
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        return mean_mag > 1.5, mean_mag


def persist_snapshot(camera_id: str, ts: str, img) -> str:
    raw_key = f"snapshots/{camera_id}/{ts.replace(':','-')}.jpg"
    latest_key = f"snapshots/{camera_id}/latest.jpg"
    body = encode_jpeg(img)
    s3.put_object(Bucket=SNAPSHOT_BUCKET, Key=raw_key, Body=body, ContentType='image/jpeg')
    s3.put_object(Bucket=SNAPSHOT_BUCKET, Key=latest_key, Body=body, ContentType='image/jpeg')
    return f"s3://{SNAPSHOT_BUCKET}/{raw_key}"


def build_prompt(event: Dict[str, Any], cv: Dict[str, Any], motion: Tuple[bool, float]) -> str:
    anomaly, motion_mag = motion
    camera_id = event.get('camera_id', 'UNKNOWN')
    location = event.get('location', 'UNKNOWN')
    max_cell = max((max(row) for row in cv.get('grid_counts', [[0]])), default=0)
    sys_msg = (
        "You are CrowdGuard, an autonomous safety agent. Input is structured CV output. "
        "Minimize false positives. Tools: save_snapshot(camera_id,timestamp), send_alert(level,message,recipients). "
        "Output JSON keys: action (none|save_snapshot|send_alert|multi), level (INFO|WARN|CRITICAL), message, recipients, also(optional)."
    )
    prompt = f"""
System: {sys_msg}
Input:
camera_id={camera_id}, location="{location}", people_count={cv.get('people_count')}, max_cell={max_cell}, anomaly={anomaly} (conf={0.8 if anomaly else 0.1}), motion_mean={motion_mag}
Answer format: JSON with keys: action, level, message, recipients, also(optional)
"""
    return prompt


def call_bedrock(prompt: str) -> str:
    body = {"inputText": prompt}
    resp = bedrock_runtime.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
    payload = resp.get('body')
    text = None
    if hasattr(payload, 'read'):
        payload = json.loads(payload.read())
    else:
        payload = json.loads(payload)
    text = payload.get('outputText') or payload.get('results', [{}])[0].get('outputText') or json.dumps(payload)
    return text


def call_bedrock_agent(prompt: str, session_id: str) -> str:
    # Minimal AgentCore call; falls back to model if not configured
    if not (BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID):
        return call_bedrock(prompt)
    try:
        resp = bedrock_agent_runtime.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=prompt
        )
        # Some SDKs stream tokens under 'completion'; concatenate
        chunks = []
        stream = resp.get('completion') or resp.get('responseStream')
        if stream and hasattr(stream, '__iter__'):
            for evt in stream:
                chunk = evt.get('bytes') or evt.get('chunk') or b''
                if hasattr(chunk, 'read'):
                    chunks.append(chunk.read().decode('utf-8', errors='ignore'))
                elif isinstance(chunk, (bytes, bytearray)):
                    chunks.append(chunk.decode('utf-8', errors='ignore'))
        output = ''.join(chunks) if chunks else json.dumps(resp)
        return output
    except Exception:
        return call_bedrock(prompt)


def parse_decision(text: str) -> Dict[str, Any]:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return {"action": "none", "level": "INFO", "message": "Unable to parse LLM output", "recipients": []}


def maybe_send_alert(level: str, message: Dict[str, Any]):
    if not ALERT_TOPIC_ARN:
        return
    sns.publish(TopicArn=ALERT_TOPIC_ARN, Subject=f"CrowdGuard {level}", Message=json.dumps(message))


def handle_event(event: Dict[str, Any]) -> Dict[str, Any]:
    camera_id = event.get('camera_id', 'UNKNOWN')
    ts = event.get('timestamp') or datetime.utcnow().isoformat() + 'Z'

    # Decode snapshot (raw bytes if using SageMaker endpoint)
    img = None
    if 'snapshot_b64' in event and event['snapshot_b64']:
        img = decode_image(event['snapshot_b64'])
    else:
        raise ValueError('snapshot_b64 is required for inference-only pipeline')

    # Persist snapshot and compute motion
    prev_gray = get_prev_snapshot_gray(camera_id)
    snapshot_uri = persist_snapshot(camera_id, ts, img)
    anomaly, motion_mag = compute_motion(prev_gray, img)

    # CV detection
    import time as _t
    _t0 = _t.time()
    cv_out = run_cv(img)
    cv_latency_ms = int((_t.time() - _t0) * 1000)
    if SAGEMAKER_ENDPOINT_NAME:
        # inference.py already includes motion and anomaly
        cv_out.setdefault('anomaly', anomaly)
        cv_out.setdefault('anomaly_confidence', 0.8 if anomaly else 0.1)
        cv_out.setdefault('motion_magnitude', motion_mag)
    else:
        cv_out['anomaly'] = anomaly
        cv_out['anomaly_confidence'] = 0.8 if anomaly else 0.1
        cv_out['motion_magnitude'] = motion_mag

    # LLM decision
    prompt = build_prompt(event, cv_out, (anomaly, motion_mag))
    _t1 = _t.time()
    session_id = f"{camera_id}-{ts}"
    raw = call_bedrock_agent(prompt, session_id)
    llm_latency_ms = int((_t.time() - _t1) * 1000)
    decision = parse_decision(raw)

    actions = [decision.get('action', 'none')] + ([decision.get('also')] if decision.get('also') else [])
    flat_actions = []
    for a in actions:
        if not a:
            continue
        if isinstance(a, list):
            flat_actions.extend(a)
        else:
            flat_actions.append(a)

    outputs = []
    for action in flat_actions:
        if action == 'send_alert':
            msg = {
                'level': decision.get('level', 'INFO'),
                'message': decision.get('message', ''),
                'camera_id': camera_id,
                'location': event.get('location', 'UNKNOWN'),
                'timestamp': ts,
                'snapshot': snapshot_uri,
                'cv': cv_out
            }
            maybe_send_alert(decision.get('level', 'INFO'), msg)
            outputs.append({'action': 'send_alert', 'status': 'ok'})
        elif action == 'save_snapshot':
            outputs.append({'action': 'save_snapshot', 'uri': snapshot_uri})
        else:
            outputs.append({'action': action, 'status': 'ignored'})

    # Emit CloudWatch custom metrics
    try:
        cloudwatch.put_metric_data(
            Namespace='CrowdGuard',
            MetricData=[
                {'MetricName': 'CVLatencyMs', 'Unit': 'Milliseconds', 'Value': cv_latency_ms},
                {'MetricName': 'LLMLatencyMs', 'Unit': 'Milliseconds', 'Value': llm_latency_ms},
                {'MetricName': 'PeopleCount', 'Unit': 'Count', 'Value': float(cv_out.get('people_count', 0))},
            ]
        )
    except Exception:
        pass

    # Store event in DynamoDB if configured
    if dynamodb:
        try:
            table = dynamodb.Table(EVENTS_TABLE)
            item = {
                'event_id': f"{camera_id}#{ts}",
                'camera_id': camera_id,
                'timestamp': ts,
                'location': event.get('location', 'UNKNOWN'),
                'decision': decision,
                'snapshot': snapshot_uri,
                'cv': cv_out,
            }
            table.put_item(Item=item)
        except Exception:
            pass

    return {'decision': decision, 'outputs': outputs, 'snapshot': snapshot_uri, 'cv': cv_out, 'latency_ms': {'cv': cv_latency_ms, 'llm': llm_latency_ms}}


def lambda_handler(event, context):
    # SNS trigger supported
    if 'Records' in event and event['Records'] and event['Records'][0].get('EventSource') == 'aws:sns':
        results = []
        for rec in event['Records']:
            msg = rec['Sns']['Message']
            payload = json.loads(msg)
            results.append(handle_event(payload))
        return {'results': results}

    # Direct invoke with JSON
    return handle_event(event)
