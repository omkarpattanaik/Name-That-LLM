with open("comp_stack.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines in comp_stack.py: {len(lines)}")
for i, line in enumerate(lines):
    if line.startswith("# =="):
        print(f"Line {i:4d}: {line}")

