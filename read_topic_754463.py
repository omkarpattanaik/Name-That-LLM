import kaggle
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Show topic 754463
try:
    topic_data = kaggle.api.forums_topic_show(754463)
    print("=== TOPIC 754463 ===")
    print("Title:", getattr(topic_data, 'title', ''))
    print("Body:", getattr(topic_data, 'content', getattr(topic_data, 'body', '')))
    comments = getattr(topic_data, 'comments', [])
    print(f"Comments ({len(comments)}):")
    for i, c in enumerate(comments):
        print(f"--- Comment {i} by {getattr(c, 'author', '')} ---")
        print(getattr(c, 'content', getattr(c, 'body', ''))[:1000])
except Exception as e:
    print("Error:", e)

