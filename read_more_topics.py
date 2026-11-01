import kaggle
import sys

sys.stdout.reconfigure(encoding='utf-8')

for tid in [754426, 752766, 752365, 754131, 754427]:
    t = kaggle.api.forums_topic_show(tid)
    topic_obj = t[0]
    comments = t[1]
    print("=" * 70)
    print(f"[{tid}] {getattr(topic_obj, 'title', '')}")
    print(f"Author: {getattr(topic_obj, 'author_name', getattr(topic_obj, 'authorName', ''))}")
    print(getattr(topic_obj, 'content', '')[:1200])
    print(f"--- Comments ({len(comments)}) ---")
    for c in comments:
        print(f"  * {getattr(c, 'author_name', getattr(c, 'authorName', ''))}: {getattr(c, 'content', '')[:500]}")
