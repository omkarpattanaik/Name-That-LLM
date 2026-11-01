import json
import difflib

with open('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb_omer = json.load(f)

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb_synth = json.load(f)

infer_omer = "".join(nb_omer['cells'][6]['source']).splitlines()
infer_synth = "".join(nb_synth['cells'][8]['source']).splitlines()

diff = list(difflib.unified_diff(infer_omer, infer_synth, fromfile='omer_infer', tofile='synth_infer', lineterm=''))
print(f"Diff lines: {len(diff)}")
for l in diff[:60]:
    print(l)

