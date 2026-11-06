import json
import ast
import symtable
import builtins
import sys

nb_path = 'kaggle_v20/name-that-llm-v20-sovereign-grandmaster-fusion.ipynb'
meta_path = 'kaggle_v20/kernel-metadata.json'

print("--- 1. Validating kernel-metadata.json ---")
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
assert meta['id'] == 'omkarpattanaik123/name-that-llm-v20-sovereign-grandmaster-fusion', f"Unexpected id: {meta['id']}"
assert meta['code_file'] == 'name-that-llm-v20-sovereign-grandmaster-fusion.ipynb', f"Unexpected code_file: {meta['code_file']}"
assert meta['enable_gpu'] is True
assert len(meta['dataset_sources']) == 20, f"Expected 20 datasets, got {len(meta['dataset_sources'])}"
assert all(isinstance(d, str) and d.strip() for d in meta['dataset_sources'])
assert len(meta['kernel_sources']) == 8
assert len(meta['model_sources']) == 8
print(f"Metadata valid! Datasets: {len(meta['dataset_sources'])}, Kernels: {len(meta['kernel_sources'])}, Models: {len(meta['model_sources'])}")

print("\n--- 2. Validating Notebook Syntax (ast.parse) ---")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        clean_lines = []
        for line in src.split('\n'):
            if line.strip().startswith('!') or line.strip().startswith('%'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_src = '\n'.join(clean_lines)
        try:
            ast.parse(clean_src)
        except SyntaxError as se:
            print(f"ERROR: SyntaxError in Cell {idx}: {se}")
            sys.exit(1)
print(f"All {len(nb['cells'])} cells parsed with zero syntax errors!")

print("\n--- 3. Verifying Scope and Undefined Variables (symtable) ---")
scripts = {
    'gauss.py': nb['cells'][4]['source'],
    'sibling_style.py': nb['cells'][6]['source'],
    'v34_pairwise.py': nb['cells'][8]['source'],
    'infer.py': nb['cells'][12]['source'],
    'comp_stack.py': nb['cells'][14]['source'],
}

global_defs = set(dir(builtins))

def check_undefined(code_str, filename):
    if code_str.startswith('%%writefile'):
        code_str = code_str.split('\n', 1)[1]
    top = symtable.symtable(code_str, filename, 'exec')
    file_defs = set(global_defs)
    for s in top.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace():
            file_defs.add(s.get_name())
    
    undefined = set()
    def check_table(t, path):
        for s in t.get_symbols():
            name = s.get_name()
            if s.is_global() and not (s.is_assigned() or s.is_imported() or s.is_namespace()):
                if name not in file_defs:
                    undefined.add((path, name))
        for child in t.get_children():
            check_table(child, f"{path}.{child.get_name()}")
    check_table(top, 'global')
    return undefined

all_clean = True
for name, src in scripts.items():
    code = ''.join(src)
    undef = check_undefined(code, name)
    undef = {u for u in undef if u[1] != '__file__'}
    if undef:
        print(f"WARNING: {name} has unresolved globals: {undef}")
        all_clean = False
    else:
        print(f"PASS: {name} has ZERO undefined variables!")

assert all_clean, "One or more scripts contain undefined variables!"

print("\n--- 4. Checking Calibration Parameters & Critical Fixes ---")
c2 = ''.join(nb['cells'][2]['source'])
assert 'flag_gemini37_routing_risk' in c2, "Gemini 3.7 purge missing from Cell 2"
assert 'flag_near_train_prompt' in c2, "Near-train purge missing from Cell 2"
assert 'flag_train_prompt_regen' in c2, "Train prompt regen purge missing from Cell 2"
assert 'argon_extra_pool.parquet' in c2, "POOL2 curation missing from Cell 2"
assert 'OURS_SVC_W = 4.0' in c2, "OURS_SVC_W = 4.0 not in Cell 2"
assert 'SVC_W = 8.0' in c2, "SVC_W = 8.0 not in Cell 2"
assert 'ATLAS_MARGIN = 0.32' in c2, "ATLAS_MARGIN = 0.32 not in Cell 2"
assert 'SINKHORN = 0.35' in c2, "SINKHORN = 0.35 not in Cell 2"
print("PASS: Cell 2 3-flag purge, POOL2 curation, and calibrated Dual-SVC weights confirmed!")

c6 = ''.join(nb['cells'][6]['source'])
assert 'FABLE5_SUBORDINATORS' in c6, "Specialist features missing from sibling_style.py"
assert 'prism_3window' in c6, "prism_3window missing from sibling_style.py"
assert 'logsumexp(old_logits, axis=1, keepdims=True)' in c6, "V40A residual missing from sibling_style.py"
print("PASS: Cell 6 227-feature engine & V40A residual confirmed!")

c12 = ''.join(nb['cells'][12]['source'])
assert 'import os, sys, time, gc, argparse, json, re, pathlib' in c12, "pathlib missing from infer.py"
assert '--pool2' in c12, "--pool2 arg missing from infer.py"
assert 'pool2_tfidf' in c12, "pool2_tfidf missing from infer.py"
assert 'svc4 Argon pool2 fresh rows added' in c12, "pool2 svc4 injection missing from infer.py"
assert '_marks = [' in c12, "_marks missing from infer.py"
assert '—' in c12 and '’' in c12, "Unicode glyphs corrupted in infer.py"
assert '_pairwise_path = pathlib.Path' in c12, "Pairwise path check missing"
print("PASS: Cell 12 POOL2 integration, pathlib import, & Unicode marks confirmed!")

c16 = ''.join(nb['cells'][16]['source'])
assert 'subprocess.run([sys.executable, \'comp_stack.py\']' in c16, "Cell 16 subprocess missing"
print("PASS: Cell 16 safe subprocess runner confirmed!")

c18 = ''.join(nb['cells'][18]['source'])
assert 'FAST_COMMIT SELF-TEST' in c18, "Cell 18 FAST_COMMIT self-test missing"
assert 'n14b_proven' in c18, "Tri-Adapter n14b_proven missing from Cell 18"
assert 'adapter_weights' in c18, "Tri-Adapter weights missing from Cell 18"
assert "'--pool2', str(POOL2)" in c18, "cmd pool2 argument missing from Cell 18"
assert "'--pool2_w', '0.35'" in c18, "cmd pool2_w missing from Cell 18"
print("PASS: Cell 18 Tri-Adapter 14B, POOL2 execution args, & FAST_COMMIT smoke test confirmed!")

print("\n=== V20 VERIFICATION COMPLETE: ALL CHECKS PASSED PERFECTLY! ===")
