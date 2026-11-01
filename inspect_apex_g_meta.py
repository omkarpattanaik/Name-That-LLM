import json

with open('reference_kernels/bgunshui_ntl-apex-g/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("Title:", meta.get('title'))
print("GPU:", meta.get('enable_gpu'), meta.get('machine_shape'))
print("\nDataset sources:")
for d in meta.get('dataset_sources', []):
    print(" ", d)
print("\nKernel sources:")
for k in meta.get('kernel_sources', []):
    print(" ", k)
print("\nModel sources:")
for m in meta.get('model_sources', []):
    print(" ", m)

