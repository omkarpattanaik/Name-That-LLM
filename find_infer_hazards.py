with open("infer_v5_debug.py", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if any(k in line for k in ['assert', 'raise', 'open(', '.index(', 'np.load', 'pd.read_']):
            print(f"{i:4d}: {line.strip()[:100]}")

