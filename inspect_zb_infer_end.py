import json

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

c6 = "".join(nb_zb['cells'][6]['source'])
lines = c6.splitlines()
print(f"Zbigniew infer.py total lines: {len(lines)}")
for i in range(len(lines) - 50, len(lines)):
    print(f"{i}: {lines[i]}")

