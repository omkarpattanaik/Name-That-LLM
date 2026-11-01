import json
import sys

def inspect_notebook(path):
    print("=" * 60)
    print(f"Inspecting: {path}")
    print("=" * 60)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        lines = cell.get('source', [])
        first_line = lines[0].strip() if lines else "<empty>"
        print(f"Cell {i:2d} [{cell['cell_type']:8s}] ({len(lines):4d} lines): {first_line[:80]}")

inspect_notebook('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb')
inspect_notebook('reference_kernels/alex_0662/name-that-llm-0-662-all-public-what-failed.ipynb')
inspect_notebook('reference_kernels/lightning_0661/0-661-qwen3-8-27b-lora-spectral-ensemble.ipynb')
inspect_notebook('reference_kernels/avazaku_0651/name-that-llm-0-651-verified-reproduct.ipynb')

