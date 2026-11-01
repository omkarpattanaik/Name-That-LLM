import kaggle
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Fetching all competition topics...")
resp = kaggle.api.competition_list_topics('name-that-llm')
topics = resp.topics

print(f"Total topics returned: {len(topics)}")
all_topics_meta = []
for t in topics:
    all_topics_meta.append({
        'id': t.id,
        'title': t.title,
        'author': getattr(t, 'author_name', getattr(t, 'authorName', '')),
        'comments': getattr(t, 'comment_count', getattr(t, 'commentCount', getattr(t, 'total_comments', 0))),
        'score': getattr(t, 'score', getattr(t, 'votes', 0))
    })

with open('all_topics.json', 'w', encoding='utf-8') as f:
    json.dump(all_topics_meta, f, indent=2)

print(f"Saved {len(all_topics_meta)} topics to all_topics.json")
for t in all_topics_meta[:35]:
    print(f"[{t['id']:7d}] (c:{t['comments']:2d}, s:{t['score']:2d}) {t['title']}")

