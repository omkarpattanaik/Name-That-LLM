import json

with open('all_key_discussions.json', encoding='utf-8') as f:
    disc = json.load(f)

print(f"Total discussions stored: {len(disc)}")
for i, d in enumerate(disc):
    title = d.get('title', '')
    author = d.get('author', '')
    comments_cnt = len(d.get('comments', []))
    print(f"[{i:2d}] {title} | Author: {author} | Comments: {comments_cnt}")

