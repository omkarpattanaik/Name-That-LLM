import json

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 10 is infer.py
c10 = "".join(nb['cells'][10]['source'])
if c10.startswith("%%writefile infer.py\n"):
    infer_py = c10[len("%%writefile infer.py\n"):]
else:
    infer_py = c10

with open("test_infer.py", "w", encoding="utf-8") as f:
    f.write(infer_py)

print("Original test_infer.py length:", len(infer_py))

# Now run the patch from Cell 11
_p = 'test_infer.py'
_src = open(_p, encoding='utf-8').read()

_a_old = "p.add_argument('--pool', default='')"
_a_new = _a_old + ("\np.add_argument('--pool2', default=''); p.add_argument('--pool2_w', type=float, default=0.35); "
                   "p.add_argument('--pool2_tfidf', type=float, default=0.0)")
assert _src.count(_a_old) == 1, 'argparse anchor not found'
_src = _src.replace(_a_old, _a_new)

_s_old = "        _P = _P.loc[_keep].copy(); _parts.append(_P)"
_s_new = _s_old + """
    if getattr(a, 'pool2', '') and os.path.exists(a.pool2):
        _Q = pd.read_parquet(a.pool2)
        if not {'text', 'model'}.issubset(_Q.columns):
            raise ValueError(f'Pool2 must contain text/model columns: {a.pool2}')
        _Q = _Q[_Q.model.isin(LABELS)][['text', 'model']].copy()
        _Q['text'] = _Q.text.fillna('').astype(str); _Q['_srcw'] = float(getattr(a, 'pool2_w', 0.35))
        _keep = []
        for _t in _Q.text.values:
            _k = _norm_text_key(_t); _ok = bool(_k) and _k not in _seen
            _keep.append(_ok)
            if _ok: _seen.add(_k)
        _Q = _Q.loc[_keep].copy(); _parts.append(_Q)
        log(f'svc4 pool2 rows={len(_Q)} weight={float(getattr(a, "pool2_w", 0.35))}')"""
assert _src.count(_s_old) == 1, 'svc pool anchor not found'
_src = _src.replace(_s_old, _s_new)

_f_old = "FIT_W = np.concatenate([np.ones(len(train), dtype=np.float32), np.full(len(EXT_T), a.extra_w, dtype=np.float32)])"
_f_new = _f_old + """
if getattr(a, 'pool2', '') and os.path.exists(a.pool2) and float(getattr(a, 'pool2_tfidf', 0.0)) > 0:
    _Q = pd.read_parquet(a.pool2)
    _Q = _Q[_Q.model.isin(LABELS)][['text', 'model']].copy()
    _Q['text'] = _Q.text.fillna('').astype(str)
    FIT_T = np.concatenate([FIT_T, _Q.text.values])
    FIT_Y = np.concatenate([FIT_Y, _Q.model.map(L2I).values.astype(int)])
    FIT_W = np.concatenate([FIT_W, np.full(len(_Q), float(a.pool2_tfidf), dtype=np.float32)])
    log(f'ngram members pool2 rows={len(_Q)} weight={float(a.pool2_tfidf)}')"""
assert _src.count(_f_old) == 1, 'fit anchor not found'
_src = _src.replace(_f_old, _f_new)

_e_old = "    _evidence = np.clip(1.5 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)"
_e_new = "    _evidence = np.clip(1.2 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)"
assert _src.count(_e_old) == 1, "evidence anchor not found"
_src = _src.replace(_e_old, _e_new)

open(_p, "w", encoding="utf-8").write(_src)
print("Patched test_infer.py successfully, new length:", len(_src))

# Verify Python AST syntax of patched file
import ast
ast.parse(_src)
print("Patched infer.py AST syntax: 100% VALID!")

