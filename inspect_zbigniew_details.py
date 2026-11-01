import json

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell 2 ===")
print("".join(nb["cells"][2]["source"]))

c6 = "".join(nb["cells"][6]["source"])
print("=== Cell 6 svc4 section ===")
lines = c6.splitlines()
svc4_start = -1
for i, l in enumerate(lines):
    if "svc4" in l:
        svc4_start = i
        break
if svc4_start != -1:
    for i in range(svc4_start, min(len(lines), svc4_start + 60)):
        print(f"{i}: {lines[i]}")

print("=== Cell 12 ===")
print("".join(nb["cells"][12]["source"]))

