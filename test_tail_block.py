import pandas as pd
import numpy as np
import scipy.special as _v38sp
import json as _v38json
import re, pathlib
from sibling_style import apply_grouped_style_residual

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y = train.model.map(L2I).values
Gtr = train.prompt_id.values

# Simulate raw logits
_raw = np.zeros((len(test), len(LABELS)), dtype=np.float64)
_v38_before_hard = _raw.copy()
_out = pathlib.Path('test_out.parquet')

_marks = [' \u2014 ', '\u2014', '\u2013', '\u2019', '\u2018', '\u201c', '\u201d', '\u2026', '"', "'", ';', ':', '(', ')', '[', ']', '**', '`', '/', '\\', '\n', '\n\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ', '\n- ', '\n1.', '###', '```', '> ']
_Xtr = np.array([[float(_m in _t) for _m in _marks] for _t in train.text.values], dtype=np.float32)
_Xte = np.array([[float(_m in _t) for _m in _marks] for _t in test.text.values], dtype=np.float32)
_freq = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
_hits = np.stack([_Xtr[y == _k].sum(axis=0) for _k in range(len(LABELS))], axis=0).astype(np.float64)
_global_rate = (_Xtr.sum(axis=0) + 2.0) / (len(_Xtr) + 4.0)
_kappa = 24.0
_rate = (_hits + _kappa * _global_rate[None, :]) / (_freq[:, None] + _kappa)
_rate = np.clip(_rate, 0.01, 0.99)
_global_rate = np.clip(_global_rate, 0.01, 0.99)
_log_rate = np.log(_rate); _log_not_rate = np.log1p(-_rate)
_base_log_rate = np.log(_global_rate); _base_log_not_rate = np.log1p(-_global_rate)
_signal = np.sqrt(np.mean((_rate - _global_rate[None, :]) ** 2, axis=0))
_fw = _signal / max(float(np.median(_signal[_signal > 0])) if np.any(_signal > 0) else 1.0, 1e-6)
_fw = np.clip(_fw, 0.25, 2.0)
_evidence = (_Xte * _fw[None, :]) @ _log_rate.T + ((1.0 - _Xte) * _fw[None, :]) @ _log_not_rate.T
_evidence -= (((_Xte * _fw[None, :]) @ _base_log_rate) + (((1.0 - _Xte) * _fw[None, :]) @ _base_log_not_rate))[:, None]
_evidence -= _evidence.mean(axis=1, keepdims=True)
_evidence = np.clip(1.5 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)
_raw = _v38_before_hard.copy()
_probs = _v38sp.softmax(_raw, axis=1)
_sorted = np.sort(_probs, axis=1)
_margin = _sorted[:, -1] - _sorted[:, -2]
_confidence_gate = np.clip((0.35 - _margin) / 0.35, 0.0, 1.0)
_word_count = np.array([len(re.findall(r'\b\w+\b', str(_t))) for _t in test.text.values], dtype=np.float32)
_length_gate = np.clip(np.sqrt(np.maximum(_word_count, 1.0) / 80.0), 0.40, 1.0)
_active_markers = _Xte.sum(axis=1)
_coverage_gate = np.clip(np.sqrt((_active_markers + 1.0) / 4.0), 0.45, 1.0)
_gate = _confidence_gate * _length_gate * _coverage_gate
_safe_logits = _raw + _gate[:, None] * _evidence

_reference_logits = _safe_logits.copy()
_v31_sub = lambda zz: pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in zz.argmax(axis=1)]})
_v31_sub(_reference_logits).to_parquet(_out.with_name('candidate_reference_0660.parquet'), index=False)

print("Running sibling_style...")
_safe_logits = apply_grouped_style_residual(
    _safe_logits, train.text.values, test.text.values, LABELS, y, Gtr,
    max_alpha=0.08, diagnostics_path=str(_out.with_name('v40a_style_oof.json')))
print("sibling_style success!")
_safe = _v31_sub(_safe_logits)
print(_safe.head())

