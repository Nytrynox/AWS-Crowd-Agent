import json

def decide(cv):
    count = int(cv.get('people_count', 0) or 0)
    anomaly = bool(cv.get('anomaly', False))
    conf = float(cv.get('anomaly_confidence', 0.0) or 0.0)
    grid = cv.get('grid_counts', [[0]])
    max_grid = max(max(row) for row in grid)

    if count > 200 and anomaly and conf > 0.7:
        return {
            'action': 'send_alert',
            'level': 'CRITICAL',
            'message': f'High crowd ({count}) + abnormal motion. Possible incident.',
            'recipients': [] ,
            'also': ['save_snapshot']
        }
    elif count > 120:
        return {
            'action': 'send_alert',
            'level': 'WARN',
            'message': f'High density observed (peak cell={max_grid}). Monitoring.',
            'recipients': []
        }
    else:
        return {
            'action': 'none',
            'level': 'INFO',
            'message': 'Normal activity',
            'recipients': []
        }


if __name__ == '__main__':
    import sys
    cv_json = json.loads(sys.stdin.read())
    print(json.dumps(decide(cv_json)))
