import json, glob, os

for path in glob.glob('reference_kernels/*/*.ipynb'):
    if any(k in path for k in ['bgunshui', 'hardikgautam', 'afantihust']):
        print(f"\n=======================================================")
        print(f"File: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            print(f"Cells count: {len(nb['cells'])}")
            c0 = "".join(nb['cells'][0]['source'])
            print(f"Cell 0: {c0[:200].replace(chr(10), ' ')}")
            # Find the run command cell (usually the last code cell)
            for c in reversed(nb['cells']):
                if c['cell_type'] == 'code':
                    src = "".join(c['source'])
                    if 'infer.py' in src:
                        for line in src.splitlines():
                            if any(k in line for k in ['--svc_w', '--distort_w', '--sinkhorn', '--atlas_w', 'MODELS =']):
                                print(f"  {line.strip()}")
                        break
        except Exception as e:
            print(f"Error reading {path}: {e}")

