import kaggle, json

raw = kaggle.api.kernels_logs('omkarpattanaik123/name-that-llm-v6-apex-robust-ensemble')
if isinstance(raw, str):
    try:
        events = json.loads(raw)
    except:
        events = [raw]
else:
    events = raw

print(f"Total log events: {len(events)}")
for e in events[-20:]:
    if isinstance(e, dict):
        d = e.get('data', '').rstrip()
        if d:
            print(f"{e.get('stream_name')}: {d}")
    else:
        print(str(e)[:100])

