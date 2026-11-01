import time, os, gc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack as _hs

# Test with 15k rows vs 30k rows
for n_rows in [15000, 30000]:
    t0 = time.time()
    texts = ["This is a sample LLM generated text with diverse vocabulary and punctuation! " * 5 for _ in range(n_rows)]
    labels = np.random.randint(0, 18, size=n_rows)
    weights = np.ones(n_rows, dtype=np.float32)
    
    print(f"\n--- Testing {n_rows} rows ---")
    _A = []
    # 4 views
    for view, lo, hi, mf in [("char", 3, 5, 80000), ("word", 1, 3, 50000), ("shape", 2, 6, 20000), ("function", 1, 3, 15000)]:
        v = TfidfVectorizer(analyzer="char" if view in ["char", "shape"] else "word", ngram_range=(lo, hi), max_features=mf, dtype=np.float32)
        _A.append(v.fit_transform(texts))
    
    X = _hs(_A, format='csr')
    del _A, texts
    gc.collect()
    print(f"CSR Matrix shape: {X.shape}, nonzeros: {X.nnz}")
    
    svc = LinearSVC(C=0.5, random_state=42, max_iter=200, class_weight='balanced')
    svc.fit(X, labels, sample_weight=weights)
    print(f"LinearSVC fit completed, Time taken: {time.time() - t0:.1f}s")
    
    del X, svc, labels, weights
    gc.collect()
