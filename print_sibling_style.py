import json

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 6 is sibling_style.py
print("=== Sibling Style ===")
print("".join(nb['cells'][6]['source']))

# Check Cell 14 (command execution)
print("=== Cell 14 ===")
print("".join(nb['cells'][14]['source']))

