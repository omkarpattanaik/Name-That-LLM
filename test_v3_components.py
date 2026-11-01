import re
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.special import logsumexp, softmax
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import sys

sys.stdout.reconfigure(encoding='utf-8')

TYPO_MARKERS = [
    ',', '.', ';', ':', '!', '?',
    '\u2019',  # ’
    '\u2018',  # ‘
    '\u201c',  # “
    '\u201d',  # ”
    '\u2014',  # —
    ' \u2014 ', #  — 
    '\u2013',  # –
    '\u2026',  # …
    '"', "'", '(', ')', '[', ']', '{', '}', '*', '`',
    '/', '\\', '\n', '\n\n', '\n- ', '\n1.', '\n2.', '###', '##', '```', '**', '> ', '\t',
    ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', 'First,', 'Finally,',
    ' - ', ': ', ' ; ', '  '
]

WORDS = [
    'the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its',
    'they','their','there','this','these','those','however','therefore','moreover','although','because',
    'perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as',
    'so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all',
    'any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who'
]

WRE = re.compile(r"[A-Za-z]+(?:['\u2019\u2018][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')

@lru_cache(maxsize=15000)
def row(t):
    t = str(t)
    s = WRE.findall(t)
    lo = [z.lower() for z in s]
    n = max(len(s), 1)
    nt = max(len(t), 1)
    ns = max(len(t.splitlines()), 1)
    lens = np.array([len(z) for z in s] or [0])
    sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    
    base = [
        np.log1p(n), np.log1p(nt), nt / n, ns / n, len(set(lo)) / n,
        float(np.mean(lens)), float(np.std(lens)),
        float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)),
        float(np.mean(lens >= 8)),
        sum(c.isupper() for c in t) / nt,
        sum(c.isdigit() for c in t) / nt,
        sum(ord(c) > 127 for c in t) / nt,
        sum(c.isspace() for c in t) / nt,
        sum(c == '\n' for c in t) / nt
    ]
    base += [min(t.count(m) / n, 3.0) for m in TYPO_MARKERS]
    cnt = {}
    for w in lo:
        cnt[w] = cnt.get(w, 0) + 1
    base += [cnt.get(w, 0) / n for w in WORDS]
    return np.array(base, dtype=np.float32)

def features(texts):
    return np.stack([row(str(t)) for t in texts])

# Test on train.csv
train = pd.read_csv('data/train.csv')
print("Testing feature extraction on 100 rows...")
feats = features(train['text'].iloc[:100].values)
print(f"Features successfully extracted, shape: {feats.shape}")
print(f"Any NaN: {np.isnan(feats).any()}, Any Inf: {np.isinf(feats).any()}")
print("Feature extraction test PASSED cleanly!")

