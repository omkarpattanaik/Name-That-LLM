import json

with open("kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 6 is sibling_style.py
c6 = "".join(nb['cells'][6]['source'])
if c6.startswith("%%writefile sibling_style.py\n"):
    c6 = c6[len("%%writefile sibling_style.py\n"):]
with open("sibling_style.py", "w", encoding="utf-8") as f:
    f.write(c6)
print("Wrote sibling_style.py, length:", len(c6))

# Cell 8 is infer.py
c8 = "".join(nb['cells'][8]['source'])
if c8.startswith("%%writefile infer.py\n"):
    c8 = c8[len("%%writefile infer.py\n"):]
with open("infer_v5_debug.py", "w", encoding="utf-8") as f:
    f.write(c8)
print("Wrote infer_v5_debug.py, length:", len(c8))

