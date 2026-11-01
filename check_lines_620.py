with open("comp_stack.py", encoding="utf-8") as f:
    lines = f.readlines()
for i in range(615, 645):
    print(f"{i:3d}: {lines[i]}", end="")

