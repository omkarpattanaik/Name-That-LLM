import kaggle
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

topic_ids = [
    754463, 752766, 753177, 754427, 754426, 752365, 754131,
    753974, 754084, 754026, 753779, 753526, 750664, 753053
]

results = {}
for tid in topic_ids:
    try:
        t = kaggle.api.forums_topic_show(tid)
        topic_obj = t[0]
        comments = t[1]
        
        parsed_comments = []
        for c in comments:
            parsed_comments.append({
                'author': str(getattr(c, 'author_name', getattr(c, 'authorName', ''))),
                'content': str(getattr(c, 'content', getattr(c, 'body', ''))),
                'postDate': str(getattr(c, 'post_date', getattr(c, 'postDate', '')))
            })
            
        results[tid] = {
            'id': tid,
            'title': str(getattr(topic_obj, 'title', '')),
            'author': str(getattr(topic_obj, 'author_name', getattr(topic_obj, 'authorName', ''))),
            'content': str(getattr(topic_obj, 'content', getattr(topic_obj, 'body', ''))),
            'comments': parsed_comments
        }
        print(f"Loaded topic {tid}: {results[tid]['title']} ({len(parsed_comments)} comments)")
    except Exception as e:
        print(f"Error loading topic {tid}: {e}")

with open('all_key_discussions.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=str)

print("Saved all discussions to all_key_discussions.json")
