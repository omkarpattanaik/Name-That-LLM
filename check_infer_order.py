with open("infer_v5_debug.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
for i, line in enumerate(lines):
    if line.startswith("# ----------------") or line.startswith("if a."):
        print(f"Line {i:3d}: {line}")

