with open("comp_stack.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
for i in range(len(lines) - 40, len(lines)):
    print(f"{i:3d}: {lines[i]}")

