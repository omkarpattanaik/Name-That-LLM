import json

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print("Zbigniew cells count:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    for kw in ["open_pool", "LinearSVC", "svc_w", "POOL", "fit_transform", "svc4"]:
        if kw in src:
            print(f"=== Cell {i} (contains {kw}) ===")
            print(src[:600])
            print("...\n")
            break

