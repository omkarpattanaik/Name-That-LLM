import json, os

def check_kernel(path):
    print("=" * 60)
    print("Checking:", path)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells']):
        txt = "".join(cell.get('source', []))
        lines = txt.splitlines()
        first = lines[0] if lines else ""
        if 'score' in txt.lower() or 'lb' in txt.lower() or '0.6' in txt or '0.7' in txt or 'w_' in txt or 'cmd' in txt or 'infer.py' in txt:
            print(f"Cell {i} [{cell['cell_type']}]: {first[:80]}")
            for l in lines:
                if any(k in l.lower() for k in ['lb', 'score', 'cmd =', 'sinkhorn', 'twin', 'claude', 'atlas_w', 'qwen', '0.6', '0.7']):
                    print("   -->", l.strip()[:100])

for p in [
    'reference_kernels/judith_0664/0-664-hyperparameter-tuned-who-are-u.ipynb',
    'reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb',
    'reference_kernels/qasmi_v5/name-that-llm-v5.ipynb',
]:
    if os.path.exists(p):
        check_kernel(p)

