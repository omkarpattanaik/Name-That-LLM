with open("comp_stack.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
for i in range(1, 135):
    print(f"{i:3d}: {lines[i]}")

