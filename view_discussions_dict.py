import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('all_key_discussions.json', encoding='utf-8') as f:
    disc = json.load(f)

for tid, t in disc.items():
    print(f"=== Topic {tid}: {t.get('title')} ===")
    print(f"Author: {t.get('author')}")
    body = t.get('content', '')
    # Strip HTML tags simply
    import re
    clean_body = re.sub(r'<[^>]+>', ' ', body)
    clean_body = ' '.join(clean_body.split())
    print(f"Content: {clean_body[:400]}...")
    comments = t.get('comments', [])
    print(f"Comments count: {len(comments)}")
    for c in comments[:3]:
        c_body = re.sub(r'<[^>]+>', ' ', c.get('content', ''))
        c_body = ' '.join(c_body.split())
        print(f"  - [{c.get('author')}]: {c_body[:200]}...")
    print("-" * 60)
