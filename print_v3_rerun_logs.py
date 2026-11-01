import kaggle
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs = kaggle.api.kernels_logs('omkarpattanaik123/name-that-llm-v3-deep-frontier-ensemble')
print(f"Total log events: {len(logs)}")
for l in logs[-60:]:
    data = l.get('data', '').rstrip()
    if data:
        print(data)

