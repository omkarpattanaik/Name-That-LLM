import kaggle, os

target_kernels = [
    'bgunshui/ntl-apex-h',
    'bgunshui/ntl-apex-g',
    'hardikgautam25/two-stage-lora-pool-champion',
    'afantihust/ntl-fork664-s50',
    'afantihust/ntl-fork664-m25'
]

for ref in target_kernels:
    out_dir = os.path.join('reference_kernels', ref.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)
    try:
        kaggle.api.kernels_pull(ref, path=out_dir)
        print(f"Pulled {ref} -> {out_dir}")
    except Exception as e:
        print(f"Failed to pull {ref}: {e}")

