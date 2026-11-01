import kaggle
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs_raw = kaggle.api.kernels_logs('omkarpattanaik123/name-that-llm-v3-deep-frontier-ensemble')
if isinstance(logs_raw, str):
    try:
        events = json.loads(logs_raw)
    except:
        events = [logs_raw]
else:
    events = logs_raw

print(f"Events type: {type(events)}, len: {len(events)}")
for e in events[-40:]:
    if isinstance(e, dict):
        print(e.get('data', '').rstrip())
    else:
        print(str(e)[:100])

