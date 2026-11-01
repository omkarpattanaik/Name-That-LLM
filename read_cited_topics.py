import kaggle
import sys

sys.stdout.reconfigure(encoding='utf-8')

for tid in [750390, 750706, 751544, 751011]:
    try:
        t = kaggle.api.forums_topic_show(tid)
        topic_obj = t[0]
        comments = t[1]
        print("=" * 70)
        print(f"[{tid}] {getattr(topic_obj, 'title', '')}")
        print(f"Author: {getattr(topic_obj, 'author_name', getattr(topic_obj, 'authorName', ''))}")
        print(getattr(topic_obj, 'content', '')[:1200])
        print(f"--- Comments ({len(comments)}) ---")
        for c in comments[:3]:
            print(f"  * {getattr(c, 'author_name', getattr(c, 'authorName', ''))}: {getattr(c, 'content', '')[:400]}")
    except Exception as e:
        print(f"Failed {tid}:", e)

