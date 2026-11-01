with open("comp_stack.py", encoding="utf-8") as f:
    lines = f.readlines()
for i in range(355, 385):
    print(f"{i:3d}: {lines[i]}", end="")

