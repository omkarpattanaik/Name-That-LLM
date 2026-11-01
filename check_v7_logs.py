import kaggle, json

raw = kaggle.api.kernels_logs('omkarpattanaik123/name-that-llm-v7-apex-sovereign-ensemble')
if isinstance(raw, str) and raw:
    try:
        events = json.loads(raw)
    except:
        events = [raw]
    print(f"Total log events: {len(events)}")
    for e in events[-10:]:
        if isinstance(e, dict):
            d = e.get('data', '').rstrip()
            if d:
                print(f"{e.get('stream_name')}: {d}")
else:
    print("Logs empty, container still provisioning / extracting weights")

