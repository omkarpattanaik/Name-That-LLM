import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('all_key_discussions.json', 'r', encoding='utf-8') as f:
    discussions = json.load(f)

for tid in [754026, 753177, 753053, 750664, 753974]:
    d = discussions.get(str(tid))
    if d:
        print("=" * 80)
        print(f"TOPIC {tid}: {d['title']} | Author: {d['author']}")
        print("=" * 80)
        print(d['content'])
        print(f"\n--- Comments ({len(d['comments'])}) ---")
        for c in d['comments']:
            print(f"[{c['author']}]: {c['content']}\n")

